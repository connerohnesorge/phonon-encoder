"""Reference-based voice adaptation: raw reference audio -> a Phonon voice file.

1. The speaker encoder turns the reference (conditioned as the runtime does) into a voice prompt.
2. Whisper transcribes the reference with word timestamps; it is cut into 2-4 s phrases, and the
   codec-encoder stand-in turns each phrase into Phonon codec latents.
3. The prompt is adjusted so that the frozen flow LM best predicts this speaker's own phrases
   (its flow-matching and end-of-speech loss, teacher-forced), starting from the encoder's output.
   `affine` mode learns only a per-channel scale and shift of the prompt (1,024 numbers), which
   cannot memorise the phrases' content; `full` mode learns an L2-penalised offset per frame.

Only the reference recording is used; no speaker-verification model is involved. The result is
written as a `speaker_wavs` voice file that `ptts --voice voice.safetensors` loads.

Usage: adapt_voice.py MODEL_ID OUT_DIR REF.wav [REF.wav ...]
"""
import os, sys, time

import librosa
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration, pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_spk import loud
from refcond import prepare_reference
from train_emain import MelLoss, latent_stats, load_decoder, load_emain
from train_spk import W, flow_loss, load_speaker_encoder
from train_spk_raw import chat, pad
from lmprior import LAM as LM_PRIOR, refine_lm
import pyloudnorm

MODELS = {
    "31m": ("gradium/phonon-31m-en", "92741c5b07d085ff840e26fea8ce43110b87852f"),
    "40m": ("gradium/phonon-40m-en", "cc738b7e4d182c2ba1bafb8062aa4846af000637"),
}
SR = 24000
MODE = os.environ.get("MODE", "affine")
STEPS = int(os.environ.get("STEPS", "120"))
LR = float(os.environ.get("ADAPT_LR", "0.02" if MODE == "affine" else "0.01"))
L2 = float(os.environ.get("L2", "1.0"))
REFINE_STEPS = int(os.environ.get("REFINE_STEPS", "0"))
# second stage: phrase latents refined under the LM prior (lmprior.py), prompt re-adapted on them
STEPS2 = int(os.environ.get("STEPS2", "150"))
_dec, _mel, _sd = None, None, None


def refine(z, x):
    """Sharpens a phrase's target latents: decoder(z) is pulled towards the phrase's audio
    (multi-resolution mel), starting from the codec-encoder estimate and kept near it."""
    global _dec, _mel, _sd
    if _dec is None:
        _dec, _mel, _sd = load_decoder(), MelLoss(), latent_stats()[1]
    z0 = torch.from_numpy(z)[None]
    zz = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([zz], lr=0.02)
    xt = torch.from_numpy(x)[None]
    for _ in range(REFINE_STEPS):
        y = _dec.decode(zz).audio_values[:, 0, : xt.shape[-1]]
        loss = _mel(y, xt) + 0.01 * (((zz - z0) / _sd) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return zz.detach()[0].numpy()


def phrases(a, asr, meter, emain, keep_audio=False):
    a16 = librosa.resample(a, orig_sr=SR, target_sr=16000)
    words = [w for w in asr(a16, return_timestamps="word")["chunks"] if w["timestamp"][1] is not None]
    segs, cur = [], []
    for w in words:
        cur.append(w)
        if cur[-1]["timestamp"][1] - cur[0]["timestamp"][0] >= 2.0:
            segs.append(cur)
            cur = []
    if cur and segs and cur[-1]["timestamp"][1] - cur[0]["timestamp"][0] < 1.0:
        segs[-1] += cur
    elif cur:
        segs.append(cur)
    out = []
    for s in segs:
        t0, t1 = s[0]["timestamp"][0], s[-1]["timestamp"][1]
        x = a[max(0, int((t0 - 0.05) * SR)) : int((t1 + 0.1) * SR)]
        x = np.clip(loud(x, -22.0, meter), -1, 1)
        x = np.pad(x, (0, (-len(x)) % 1920))
        with torch.no_grad():
            z = emain(torch.from_numpy(x)[None, None])[0].T.numpy()
        if REFINE_STEPS:
            z = refine(z, x)
        text = " ".join(w["text"].strip() for w in s).strip()
        if len(z) >= 8 and text:
            out.append((z, text, x) if keep_audio else (z, text))
    return out


def adapt(a, enc, emain, asr, proc, model, meter, log=None):
    with torch.no_grad():
        p0 = enc(torch.from_numpy(prepare_reference(a))[None, None])[0].T
    segs3 = phrases(a, asr, meter, emain, keep_audio=True)
    segs = [(z, t) for z, t, _ in segs3]
    if MODE == "none" or not segs:
        return p0, segs
    if MODE == "affine":
        g = torch.ones(512, requires_grad=True)
        b = torch.zeros(512, requires_grad=True)
        params, prompt, reg = [g, b], (lambda: p0 * g + b), (lambda: torch.zeros(()))
    else:
        d = torch.zeros_like(p0, requires_grad=True)
        params, prompt, reg = [d], (lambda: p0 + d), (lambda: L2 * d.square().mean())
    opt = torch.optim.Adam(params, lr=LR)
    gen = torch.Generator().manual_seed(0)

    def fit(segs, steps):
        inputs = chat(proc, segs, [len(p0)] * len(segs))
        for step in range(steps):
            p = prompt()
            lat, mask = pad([p] * len(segs))
            lf, le = flow_loss(model, inputs, lat, mask, n_noise=8, gen=gen)
            loss = lf + 0.1 * le + reg()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if log and (step == 0 or (step + 1) % 40 == 0):
                log(f"  step {step + 1}: flow {lf.item():.4f} eos {le.item():.4f}")

    fit(segs, STEPS)
    if LM_PRIOR:
        # targets the LM models well that still decode to the reference (lmprior.py)
        p = prompt().detach()
        segs = [(refine_lm(z, x, t, p, model, proc), t) for z, t, x in segs3]
        fit(segs, STEPS2)
    return prompt().detach(), segs


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    repo, rev = MODELS[sys.argv[1]]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    proc = AutoProcessor.from_pretrained(repo, revision=rev)
    model = GradiumPhononTTSForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    enc = load_speaker_encoder(os.environ.get("ENC", f"{W}/spk_enc_final.safetensors")).eval()
    emain = load_emain(os.environ.get("EMAIN", f"{W}/emain.safetensors")).eval()
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
    meter = pyloudnorm.Meter(SR)
    for ref in sys.argv[3:]:
        t = time.time()
        a, sr = sf.read(ref, dtype="float32")
        a = a.mean(1) if a.ndim > 1 else a
        a = librosa.resample(a, orig_sr=sr, target_sr=SR) if sr != SR else a
        name = os.path.basename(ref).replace("_ref.wav", "").replace(".wav", "")
        p, segs = adapt(a, enc, emain, asr, proc, model, meter, log=lambda m: print(m, flush=True))
        save_file({"speaker_wavs": p.T[None].contiguous()}, f"{out_dir}/{name}.safetensors",
                  metadata={"sig": "ba7a5a45", "epoch": "300"})
        print(f"{name}: {len(segs)} phrases, prompt {tuple(p.shape)}, {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()

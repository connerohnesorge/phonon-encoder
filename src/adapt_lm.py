"""Reference-based LM adaptation probe: fine-tune part of the 40m LM (with the adapted prompt) on the
reference's own phrases under the flow-matching loss, and track the identity of teacher-forced and
free generations of held-out utterances (as diag_tf.py) along the way.

Usage: adapt_lm.py VOICE_DIR SPK      env: TRAIN=flow|top|all  LMLR  LMSTEPS  EVAL_AT
"""
import itertools, os, sys

import librosa
import numpy as np
import pyloudnorm
import soundfile as sf
import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration, pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapt_voice import MODELS, SR, phrases
from continue_clone import speak
from diag_tf import tf_predict
from eval_clone import speaker_sets
from prep_spk import loud
from train_emain import load_decoder, load_emain
from train_spk import W, flow_loss
from train_spk_raw import chat, pad
from verifiers import Verifiers, load16

TRAIN = os.environ.get("TRAIN", "top")
LMLR = float(os.environ.get("LMLR", "3e-5"))
LMSTEPS = int(os.environ.get("LMSTEPS", "200"))
EVAL_AT = {int(x) for x in os.environ.get("EVAL_AT", "0,50,100,200").split(",")}


def trainable(model):
    if TRAIN == "flow":
        keep = lambda n: n.startswith("flow_head.")
    elif TRAIN == "top":
        keep = lambda n: n.startswith("flow_head.") or any(n.startswith(f"model.layers.{i}.") for i in (6, 7, 8)) \
            or n.startswith("model.norm.")
    else:
        keep = lambda n: not n.startswith("loss_weighting.")
    ps = []
    for n, p in model.named_parameters():
        p.requires_grad_(keep(n))
        if keep(n):
            ps.append(p)
    return ps


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    voice_dir, s = sys.argv[1], sys.argv[2]
    repo, rev = MODELS["40m"]
    proc = AutoProcessor.from_pretrained(repo, revision=rev)
    model = GradiumPhononTTSForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=torch.float32)
    dec, emain = load_decoder(), load_emain(os.environ.get("EMAIN", f"{W}/emain.safetensors")).eval()
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
    V, meter = Verifiers(), pyloudnorm.Meter(SR)
    _, held = speaker_sets()[s]
    real = [V.embed(load16(u["path"]))["ecapa"] for u in held]
    ceil = np.mean([a @ b for a, b in itertools.combinations(real, 2)])
    prompt = load_file(f"{voice_dir}/{s}.safetensors")["speaker_wavs"][0].T.float()
    a = sf.read(f"{W}/out/eval_final/{s}_ref.wav", dtype="float32")[0]
    segs = phrases(a, asr, meter, emain)
    tests = []
    for u in held[:3]:
        x = np.clip(loud(sf.read(u["path"], dtype="float32")[0], -22.0, meter), -1, 1)
        x = np.pad(x, (0, (-len(x)) % 1920))
        with torch.no_grad():
            tests.append((u["text"], emain(torch.from_numpy(x)[None, None])[0].T))

    def score(step):
        model.eval()
        res = {"tf": [], "free": []}
        for i, (text, z) in enumerate(tests):
            with torch.no_grad():
                zs = {"tf": tf_predict(model, proc, prompt, text, z, torch.Generator().manual_seed(i))}
                g, _ = speak(model, proc, prompt, "", torch.zeros(0, z.shape[1]), text, torch.Generator().manual_seed(i))
                zs["free"] = model.denormalize_latents(g)
                for k, zz in zs.items():
                    y = dec.decode(zz[None]).audio_values[0, 0].numpy()
                    if k == "free":
                        sf.write(f"{W}/out/adapt_lm_{s}_{TRAIN}_{step}_{i}.wav", y, SR)
                    e = V.embed(librosa.resample(y, orig_sr=SR, target_sr=16000))["ecapa"]
                    res[k] += [float(e @ real[j]) for j in range(len(real)) if j != i]
        print(f"{s} {TRAIN} lr {LMLR} step {step}:", {k: round(np.mean(v) / ceil, 3) for k, v in res.items()}, flush=True)
        model.train()

    params = trainable(model)
    opt = torch.optim.AdamW(params, lr=LMLR, weight_decay=0.0)
    inputs = chat(proc, segs, [len(prompt)] * len(segs))
    lat, mask = pad([prompt] * len(segs))
    gen = torch.Generator().manual_seed(0)
    if 0 in EVAL_AT:
        score(0)
    for step in range(1, LMSTEPS + 1):
        lf, le = flow_loss(model, inputs, lat, mask, n_noise=8, gen=gen)
        loss = lf + 0.1 * le
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 25 == 0:
            print(f"  step {step}: flow {lf.item():.4f} eos {le.item():.4f}", flush=True)
        if step in EVAL_AT:
            score(step)


if __name__ == "__main__":
    main()

"""Reference-based voice adaptation on generated speech.

Starts like adapt_voice.py (speaker encoder prompt, likelihood of the reference's own phrases),
then adds a term computed on what the model actually says: each step, the frozen flow LM speaks a
generic sentence from the current prompt (differentiable roll-out: one-step flow sampling, codec
decoder), and WavLM-SV compares it with the reference. The prompt moves to make the generated
speech sound like the reference speaker. Only the reference recording is used.

WavLM-SV is in the loop, so it is no longer an independent judge of the result: ECAPA and
Resemblyzer, which never see the generated speech here, are.

Usage: adapt_sv.py MODEL_ID OUT_DIR REF.wav [REF.wav ...]
"""
import os, random, sys, time

import librosa
import numpy as np
import pyloudnorm
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from safetensors.torch import save_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration, WavLMForXVector, pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapt_voice import MODELS, SR, phrases
from refcond import prepare_reference
from train_emain import load_decoder, load_emain
from train_spk import W, flow_loss, load_speaker_encoder
from train_spk_raw import chat, pad

STEPS = int(os.environ.get("STEPS", "120"))
LR = float(os.environ.get("ADAPT_LR", "0.002"))
FRAMES = int(os.environ.get("FRAMES", "45"))
W_LIK = float(os.environ.get("W_LIK", "1.0"))
L2 = float(os.environ.get("L2", "0.1"))
TEMP = float(os.environ.get("TEMP", "0.3"))
# generic sentences for the roll-outs, none of them an evaluation sentence
POOL = [
    "A large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds.",
    "The small pup gnawed a hole in the sock.",
]


class SV:
    def __init__(self):
        self.m = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

    def __call__(self, wav16):
        x = (wav16 - wav16.mean()) / (wav16.std() + 1e-7)
        e = self.m(input_values=x[None]).embeddings[0]
        return e / e.norm()


def rollout(model, proc, prompt, text, gen):
    """Speaks `text` from `prompt` with grads: returns normalised latents [FRAMES, 32]."""
    conv = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": text}]}]
    inp = proc.apply_chat_template(conv, tokenize=True, return_dict=True, add_generation_prompt=True,
                                   processor_kwargs={"audio_prompt_latents": [torch.zeros(len(prompt), 512)]})
    out = model.model(input_ids=inp["input_ids"], audio_prompt_latents=prompt[None], use_cache=True)
    h, cache = out.last_hidden_state[:, -1], out.past_key_values
    tok = torch.full((1, 1), model.config.audio_latent_token_id)
    lats = []
    for _ in range(FRAMES):
        noise = torch.randn(1, 32, generator=gen) * TEMP**0.5
        x = model.sample_latent(h, noise, 1)
        lats.append(x)
        o = model.model(input_ids=tok, audio_latents=x[:, None], past_key_values=cache, use_cache=True)
        h, cache = o.last_hidden_state[:, -1], o.past_key_values
    return torch.cat(lats)


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    repo, rev = MODELS[sys.argv[1]]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    proc = AutoProcessor.from_pretrained(repo, revision=rev)
    model = GradiumPhononTTSForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dec = load_decoder()
    enc = load_speaker_encoder(os.environ.get("ENC", f"{W}/spk_enc_final.safetensors")).eval()
    emain = load_emain(f"{W}/emain.safetensors").eval()
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
    sv = SV()
    meter = pyloudnorm.Meter(SR)
    for ref in sys.argv[3:]:
        t0 = time.time()
        a, sr = sf.read(ref, dtype="float32")
        a = a.mean(1) if a.ndim > 1 else a
        a = librosa.resample(a, orig_sr=sr, target_sr=SR) if sr != SR else a
        name = os.path.basename(ref).replace("_ref.wav", "").replace(".wav", "")
        with torch.no_grad():
            p0 = enc(torch.from_numpy(prepare_reference(a))[None, None])[0].T
            e_ref = sv(torch.from_numpy(librosa.resample(a, orig_sr=SR, target_sr=16000)))
        segs = phrases(a, asr, meter, emain)
        inputs = chat(proc, segs, [len(p0)] * len(segs)) if segs else None
        d = torch.zeros_like(p0, requires_grad=True)
        opt = torch.optim.Adam([d], lr=LR)
        gen = torch.Generator().manual_seed(0)
        rnd = random.Random(0)
        for step in range(STEPS):
            p = p0 + d
            z = rollout(model, proc, p, rnd.choice(POOL), gen)
            wav = dec.decode(model.denormalize_latents(z)[None]).audio_values[0, 0]
            e = sv(torchaudio.functional.resample(wav, SR, 16000))
            l_sv = 1 - (e * e_ref).sum()
            loss = l_sv + L2 * d.square().mean()
            if inputs is not None and W_LIK:
                lat, mask = pad([p] * len(segs))
                lf, le = flow_loss(model, inputs, lat, mask, n_noise=4, gen=gen)
                loss = loss + W_LIK * (lf + 0.1 * le)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([d], 1.0)
            opt.step()
            # keep the prompt with the best running similarity: the roll-outs are noisy
            avg = (1 - l_sv.item()) if step == 0 else 0.8 * avg + 0.2 * (1 - l_sv.item())
            if step == 0 or avg > best[0]:
                best = (avg, (p0 + d).detach().clone())
            if step == 0 or (step + 1) % 20 == 0:
                print(f"  {name} step {step + 1}: sv-sim {1 - l_sv.item():.3f} {time.time() - t0:.0f}s", flush=True)
        print(f"  {name}: best running sv-sim {best[0]:.3f}", flush=True)
        save_file({"speaker_wavs": best[1].T[None].contiguous()}, f"{out_dir}/{name}.safetensors",
                  metadata={"sig": "ba7a5a45", "epoch": "300"})
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

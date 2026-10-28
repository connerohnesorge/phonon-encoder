"""Probe: how much identity varies across sampling seeds, and whether verifier-free scores of a
candidate track it. Diagnostic only (ECAPA is computed here to measure the correlation)."""
import os, sys

import librosa
import numpy as np
import pyloudnorm
import soundfile as sf
import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration, pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapt_voice import MODELS, SR, phrases
from continue_clone import prefix_from_reference, speak
from eval_clone import TEXTS, speaker_sets
from refcond import prepare_reference
from train_emain import load_decoder, load_emain
from train_spk import W, flow_loss, load_speaker_encoder
from train_spk_raw import chat, pad
from verifiers import Verifiers, load16

torch.set_num_threads(4)
spk, voice_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
repo, rev = MODELS["31m"]
proc = AutoProcessor.from_pretrained(repo, revision=rev)
model = GradiumPhononTTSForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=torch.float32).eval()
dec, emain = load_decoder(), load_emain(f"{W}/emain.safetensors").eval()
enc = load_speaker_encoder(f"{W}/spk_enc_final.safetensors").eval()
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
V, meter = Verifiers(), pyloudnorm.Meter(SR)
ref, held = speaker_sets()[spk]
a = np.concatenate([sf.read(u["path"], dtype="float32")[0] for u in ref])[: 10 * SR]
prompt = load_file(f"{voice_dir}/{spk}.safetensors")["speaker_wavs"][0].T.float()
ptext, pz = prefix_from_reference(a, asr, emain, meter)
segs = phrases(a, asr, meter, emain)
real = [V.embed(load16(u["path"]))["ecapa"] for u in held]
with torch.no_grad():
    s_ref = enc(torch.from_numpy(prepare_reference(a))[None, None])[0].mean(1)
rows = []
for seed in range(n):
    z, k = speak(model, proc, prompt, ptext, pz, TEXTS[0], torch.Generator().manual_seed(seed))
    with torch.no_grad():
        wav = dec.decode(model.denormalize_latents(z)[None]).audio_values[0, 0].numpy()[k * 1920 :]
        e = V.embed(librosa.resample(wav, orig_sr=SR, target_sr=16000))["ecapa"]
        ecapa = float(np.mean([e @ r for r in real]))
        # (b) speaker-encoder cosine
        s = enc(torch.from_numpy(prepare_reference(wav))[None, None])[0].mean(1)
        enc_cos = float(torch.nn.functional.cosine_similarity(s, s_ref, dim=0).item())
        # (a) LM: the reference phrases' loss with the candidate as preceding audio
        cand = model.denormalize_latents(z[k:])
        tot = 0.0
        for zz, t in segs:
            inputs = chat(proc, [(np.concatenate([cand.numpy(), zz]), f"{TEXTS[0]} {t}")], [len(prompt)])
            lat, mask = pad([prompt])
            lf, _ = flow_loss(model, inputs, lat, mask, n_noise=8, gen=torch.Generator().manual_seed(0))
            tot += lf.item()
    rows.append((seed, ecapa, enc_cos, tot / len(segs)))
    print(f"seed {seed}: ecapa {ecapa:.3f} enc-cos {enc_cos:.3f} lm-loss {tot / len(segs):.4f}", flush=True)
r = np.array(rows)
print("spread ecapa", r[:, 1].min().round(3), r[:, 1].max().round(3),
      "| corr(ecapa, enc-cos)", np.corrcoef(r[:, 1], r[:, 2])[0, 1].round(2),
      "| corr(ecapa, -lm-loss)", np.corrcoef(r[:, 1], -r[:, 3])[0, 1].round(2))

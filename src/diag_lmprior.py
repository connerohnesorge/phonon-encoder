"""Diagnostic: do real-speech latents exist that are both faithful (decode to the recording) and
likely under the frozen LM? Per held-out utterance, latents start at E_main and are optimised for
  mel(decoder(z), x) + LAM * LM flow loss(z | adapted prompt, transcript),
then scored: resynth = decoder(z), tf = one-step LM predictions teacher-forced on z.

Usage: diag_lmprior.py VOICE_DIR SPK [SPK ...]     env: LAMS=0,1  OPT_STEPS=200
"""
import itertools, os, sys

import librosa
import numpy as np
import pyloudnorm
import soundfile as sf
import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapt_voice import MODELS, SR
from diag_tf import tf_predict
from eval_clone import speaker_sets
from prep_spk import loud
from train_emain import MelLoss, load_decoder, load_emain
from train_spk import W, flow_loss
from train_spk_raw import chat, pad
from verifiers import Verifiers, load16

LAMS = [float(x) for x in os.environ.get("LAMS", "0,1").split(",")]
OPT_STEPS = int(os.environ.get("OPT_STEPS", "200"))


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    voice_dir = sys.argv[1]
    repo, rev = MODELS[os.environ.get("MODEL", "40m")]
    proc = AutoProcessor.from_pretrained(repo, revision=rev)
    model = GradiumPhononTTSForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dec, mel = load_decoder(), MelLoss()
    for p in dec.parameters():
        p.requires_grad_(False)
    emain = load_emain(os.environ.get("EMAIN", f"{W}/emain.safetensors")).eval()
    V, meter = Verifiers(), pyloudnorm.Meter(SR)
    sets = speaker_sets()
    for s in sys.argv[2:]:
        _, held = sets[s]
        prompt = load_file(f"{voice_dir}/{s}.safetensors")["speaker_wavs"][0].T.float()
        real = [V.embed(load16(u["path"]))["ecapa"] for u in held]
        ceil = np.mean([a @ b for a, b in itertools.combinations(real, 2)])
        res = {(lam, k): [] for lam in LAMS for k in ("resynth", "tf")}
        for i, u in enumerate(held[:3]):
            a = np.clip(loud(sf.read(u["path"], dtype="float32")[0], -22.0, meter), -1, 1)
            a = np.pad(a, (0, (-len(a)) % 1920))
            x = torch.from_numpy(a)[None]
            with torch.no_grad():
                z0 = emain(x[None])[0].T
            inputs = chat(proc, [(z0.numpy(), u["text"])], [len(prompt)])
            lat, mask = pad([prompt])
            for lam in LAMS:
                z = z0.clone().requires_grad_(True)
                opt = torch.optim.Adam([z], lr=0.02)
                gen = torch.Generator().manual_seed(0)
                for step in range(OPT_STEPS):
                    lm = mel(dec.decode(z[None]).audio_values[:, 0, : x.shape[-1]], x)
                    loss = lm
                    if lam:
                        inputs["labels"] = z[None]
                        lf, _ = flow_loss(model, inputs, lat, mask, n_noise=4, gen=gen)
                        loss = loss + lam * lf
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                zz = z.detach()
                with torch.no_grad():
                    inputs["labels"] = zz[None]
                    lf, _ = flow_loss(model, inputs, lat, mask, n_noise=16, gen=torch.Generator().manual_seed(1))
                    print(f"  {s} utt {i} lam {lam}: mel {lm.item():.3f} lm-flow {lf.item():.3f}", flush=True)
                    zs = {"resynth": zz, "tf": tf_predict(model, proc, prompt, u["text"], zz, torch.Generator().manual_seed(i))}
                    for k, v in zs.items():
                        y = dec.decode(v[None]).audio_values[0, 0].numpy()
                        e = V.embed(librosa.resample(y, orig_sr=SR, target_sr=16000))["ecapa"]
                        res[(lam, k)] += [float(e @ real[j]) for j in range(len(real)) if j != i]
        print(s, {f"{lam}/{k}": round(np.mean(v) / ceil, 3) for (lam, k), v in res.items()}, flush=True)


if __name__ == "__main__":
    main()

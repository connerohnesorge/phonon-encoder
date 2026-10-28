"""Upper bound on speaker identity for anything Phonon outputs: its codec decoder renders every
clone, so no voice prompt can beat the best latents for a real recording.

For each eval speaker, a few held-out recordings are inverted: codec latents start from the
rebuilt codec encoder and are optimised so the frozen decoder's output matches the recording
(multi-resolution mel). The decoded audio is scored against the speaker's *other* real recordings,
as the clones are, and divided by the real same-speaker ceiling.

Usage: decoder_bound.py [N_UTTS] [STEPS]
"""
import itertools, json, os, sys

import librosa
import numpy as np
import pyloudnorm
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_clone import speaker_sets
from prep_spk import loud
from train_emain import MelLoss, load_decoder, load_emain
from verifiers import Verifiers, load16

W = os.environ.get("PHONON_WORK", "/home/user/work")

if __name__ == "__main__":
    n_utts = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    enc, dec, mel, V = load_emain(os.environ.get("EMAIN", f"{W}/emain.safetensors")).eval(), load_decoder(), MelLoss(), Verifiers()
    meter = pyloudnorm.Meter(24000)
    out = {}
    for s, (_, held) in speaker_sets().items():
        real = [V.embed(load16(u["path"])) for u in held]
        rows = {v: {"encoder": [], "optimised": []} for v in Verifiers.names}
        for i, u in enumerate(held[:n_utts]):
            a = np.clip(loud(sf.read(u["path"], dtype="float32")[0], -22.0, meter), -1, 1)[: 24000 * 6]
            a = a[: len(a) // 1920 * 1920]
            x = torch.from_numpy(a)[None]
            with torch.no_grad():
                z0 = enc(x[None]).transpose(1, 2)
            z = z0.clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=0.02)
            for _ in range(steps):
                loss = mel(dec.decode(z).audio_values[:, 0, : x.shape[-1]], x)
                opt.zero_grad()
                loss.backward()
                opt.step()
            for kind, zz in (("encoder", z0), ("optimised", z.detach())):
                with torch.no_grad():
                    y = dec.decode(zz).audio_values[0, 0].numpy()
                e = V.embed(librosa.resample(y, orig_sr=24000, target_sr=16000))
                for v in Verifiers.names:
                    rows[v][kind] += [float(e[v] @ real[j][v]) for j in range(len(real)) if j != i]
        out[s] = {}
        for v in Verifiers.names:
            ceil = np.mean([real[i][v] @ real[j][v] for i, j in itertools.combinations(range(len(real)), 2)])
            out[s][v] = {"ceiling": float(ceil), **{k: float(np.mean(r) / ceil) for k, r in rows[v].items()}}
        print(s, {v: (round(out[s][v]["encoder"], 2), round(out[s][v]["optimised"], 2)) for v in Verifiers.names}, flush=True)
    json.dump(out, open(os.environ.get("OUT_JSON", f"{W}/out/decoder_bound.json"), "w"), indent=1)
    for v in Verifiers.names:
        print(v, "mean ratio: encoder", round(np.mean([out[s][v]["encoder"] for s in out]), 3),
              "optimised", round(np.mean([out[s][v]["optimised"] for s in out]), 3))

"""How much of a speaker's identity survives Phonon's codec at all.

Each held-out real recording is round-tripped through the codec (the rebuilt encoder, then
Phonon's released decoder) and scored against the speaker's *other* real recordings, next to the
real-vs-real ceiling. A clone is rendered by the same decoder, so this approximates the most it
can reach (the rebuilt encoder adds some loss of its own).
"""
import itertools, json, os, sys

import numpy as np
import pyloudnorm
import soundfile as sf
import torch
import librosa

sys.path.insert(0, os.path.dirname(__file__))
from eval_clone import speaker_sets, EVAL
from prep_spk import loud
from train_emain import load_decoder, load_emain
from verifiers import Verifiers, load16

W = os.environ.get("PHONON_WORK", "/home/user/work")

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    enc, dec, V = load_emain(f"{W}/emain.safetensors").eval(), load_decoder(), Verifiers()
    meter = pyloudnorm.Meter(24000)
    out = {}
    for s, (_, held) in speaker_sets().items():
        real, rt = [], []
        for u in held:
            a = np.clip(loud(sf.read(u["path"], dtype="float32")[0], -22.0, meter), -1, 1)
            a = a[: len(a) // 1920 * 1920]
            with torch.no_grad():
                y = dec.decode(enc(torch.from_numpy(a)[None, None]).transpose(1, 2)).audio_values[0, 0].numpy()
            real.append(V.embed(load16(u["path"])))
            rt.append(V.embed(librosa.resample(y, orig_sr=24000, target_sr=16000)))
        out[s] = {}
        for v in Verifiers.names:
            ceil = np.mean([real[i][v] @ real[j][v] for i, j in itertools.combinations(range(len(real)), 2)])
            cod = np.mean([rt[i][v] @ real[j][v] for i in range(len(rt)) for j in range(len(real)) if i != j])
            out[s][v] = {"ceiling": float(ceil), "codec": float(cod), "ratio": float(cod / ceil)}
        print(s, {v: round(out[s][v]["ratio"], 2) for v in Verifiers.names}, flush=True)
    json.dump(out, open(f"{W}/out/codec_ceiling.json", "w"), indent=1)

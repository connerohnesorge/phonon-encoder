"""Diagnostic, in-domain: Phonon's own utterances in released catalog voices (31m, released
`speaker_wavs`, true codec latents). Does teacher-forced one-step prediction keep identity when
prompt and latents are exactly what the LM was trained on? Ratios against the voice's own
utterance-to-utterance ECAPA consistency.

Usage: diag_syn.py [N_VOICES]
"""
import collections, itertools, os, sys

import librosa
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapt_voice import MODELS, SR
from diag_tf import tf_predict
from train_emain import load_decoder
from train_spk import W
from verifiers import Verifiers


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    n_voices = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    repo, rev = MODELS["31m"]
    proc = AutoProcessor.from_pretrained(repo, revision=rev)
    model = GradiumPhononTTSForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=torch.float32).eval()
    dec, V = load_decoder(), Verifiers()
    by = collections.defaultdict(list)
    for it in np.load(f"{W}/synth/shard3.npy", allow_pickle=True):
        by[it["voice"]].append(it)
    voices = sorted(v for v, its in by.items() if len(its) >= 5)[:n_voices]
    emb = lambda y: V.embed(librosa.resample(y.astype(np.float32), orig_sr=SR, target_sr=16000))["ecapa"]
    tot = collections.defaultdict(list)
    for v in voices:
        its = by[v][:6]
        prompt = load_file(f"{W}/voices/voices/{v}.safetensors")["speaker_wavs"][0].T.float()
        real = [emb(it["audio"] / 32767) for it in its]
        ceil = np.mean([a @ b for a, b in itertools.combinations(real, 2)])
        res = collections.defaultdict(list)
        for i, it in enumerate(its[:3]):
            z = torch.from_numpy(it["latents"].astype(np.float32))
            with torch.no_grad():
                zs = {"resynth": z, "tf": tf_predict(model, proc, prompt, it["text"], z, torch.Generator().manual_seed(i))}
                for k, zz in zs.items():
                    e = emb(dec.decode(zz[None]).audio_values[0, 0].numpy())
                    res[k] += [float(e @ real[j]) for j in range(len(real)) if j != i]
        r = {k: round(np.mean(x) / ceil, 3) for k, x in res.items()}
        for k, x in r.items():
            tot[k].append(x)
        print(v, f"ceiling {ceil:.3f}", r, flush=True)
    print("mean", {k: round(np.mean(x), 3) for k, x in tot.items()})


if __name__ == "__main__":
    main()

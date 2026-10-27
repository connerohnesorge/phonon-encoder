"""Caches, per training-pool utterance:
- `feat`: the speaker encoder's frozen SEANet front on the utterance at -18 LUFS, the level the
  Rust `create_voice` path gives a reference (fp16, [512, n/960]);
- `z`: the codec-encoder stand-in's latents of the utterance at -22 LUFS, Phonon's own output
  level (fp16, [T, 32]): the teacher-forcing targets.
Usage: prep_spk.py feat|z SHARD NSHARDS
"""
import json, os, sys

import numpy as np
import pyloudnorm
import soundfile as sf
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(__file__))
from mimi_torch import EncCfg, MimiEncoder, load_moshi

W = os.environ.get("PHONON_WORK", "/home/user/work")


def loud(a, target, meter):
    a = a - a.mean()
    L = meter.integrated_loudness(a.astype(np.float64))
    if np.isfinite(L) and np.sqrt((a**2).mean()) >= 2e-3:
        a = a * 10 ** ((target - L) / 20)
    return a.astype(np.float32)


def main(what, shard, nshards):
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    meta = [json.loads(l) for l in open(f"{W}/audio/{os.environ.get('META', 'meta.jsonl')}")]
    meta = [m for m in meta if m["split"] == "train"]
    # at most MAX_PER_SPK utterances per speaker, the same ones for `feat` and `z`
    cap, seen, kept = int(os.environ.get("MAX_PER_SPK", "25")), {}, []
    for m in meta:
        seen[m["spk"]] = seen.get(m["spk"], 0) + 1
        if seen[m["spk"]] <= cap:
            kept.append(m)
    meta = kept[shard::nshards]
    meter = pyloudnorm.Meter(24000)
    if what == "feat":
        enc = MimiEncoder(EncCfg()).eval()
        load_moshi(enc, load_file(f"{W}/kmimi/moshi_tokenizer.safetensors"))
        f = lambda x: enc.seanet(x)[0]
        level = -18.0
    else:
        from train_emain import load_emain
        enc = load_emain(f"{W}/emain.safetensors").eval()
        f = lambda x: enc(x)[0].T
        level = -22.0
    # written in chunks of 500 utterances, so a restarted job resumes where it stopped
    os.makedirs(f"{W}/cache", exist_ok=True)
    with torch.no_grad():
        for c in range(0, len(meta), 500):
            # TAG keeps the chunks of another metadata file apart (still matched by `{what}_[0-9]*`)
            path = f"{W}/cache/{what}_{os.environ.get('TAG', '')}{shard}_{c // 500:03d}.npy"
            if os.path.exists(path):
                continue
            out = {}
            for m in meta[c : c + 500]:
                a, _ = sf.read(m["path"], dtype="float32")
                a = np.clip(loud(a, level, meter), -1, 1)
                out[m["path"]] = f(torch.from_numpy(a)[None, None]).numpy().astype(np.float16)
            np.save(path + ".tmp.npy", out, allow_pickle=True)
            os.replace(path + ".tmp.npy", path)
            print(what, shard, c + len(out), len(meta), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

"""Speaker-encoder SEANet features of the synthetic utterances heard through simulated recording
channels: room reverb, EQ, band-limiting, high-pass, noise floor. Variant k of every utterance of a
voice shares one channel (seeded by voice and k), as one real speaker's recordings share a setup.
Writes cache/synfeat_aug{k}.npy for k = 1..K, aligned with cache/synfeat.npy.
"""
import json, os, sys, zlib

import numpy as np
import pyloudnorm
import scipy.signal as ss
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(__file__))
from mimi_torch import EncCfg, MimiEncoder, load_moshi
from prep_spk import loud

W = os.environ.get("PHONON_WORK", "/home/user/work")
SR = 24000


def channel(seed):
    r = np.random.default_rng(seed)
    c = {"hp": r.uniform(40, 160)}
    if r.random() < 0.5:
        c["lp"] = float(r.choice([4000, 5500, 8000, 11000]))
    c["shelf"] = (r.uniform(-8, 8), r.uniform(-8, 8))  # low / high shelf gains, dB
    if r.random() < 0.6:
        t60 = r.uniform(0.12, 0.6)
        n = int(t60 * SR)
        tail = r.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
        rir = np.zeros(n)
        rir[0] = 1.0
        rir += tail * r.uniform(0.05, 0.4) / np.sqrt(np.sum(tail**2) + 1e-9)
        c["rir"] = rir
    if r.random() < 0.7:
        c["snr"] = r.uniform(15, 45)
        c["pink"] = r.random() < 0.5
        c["noise_seed"] = int(r.integers(1 << 31))
    return c


def apply(a, c):
    a = ss.sosfilt(ss.butter(2, c["hp"], "highpass", fs=SR, output="sos"), a)
    if "lp" in c:
        a = ss.sosfilt(ss.butter(8, c["lp"], "lowpass", fs=SR, output="sos"), a)
    lo, hi = c["shelf"]
    low = ss.sosfilt(ss.butter(1, 300, "lowpass", fs=SR, output="sos"), a)
    high = ss.sosfilt(ss.butter(1, 3000, "highpass", fs=SR, output="sos"), a)
    a = a + (10 ** (lo / 20) - 1) * low + (10 ** (hi / 20) - 1) * high
    if "rir" in c:
        a = ss.fftconvolve(a, c["rir"])[: len(a)]
    if "snr" in c:
        r = np.random.default_rng(c["noise_seed"])
        n = r.standard_normal(len(a))
        if c["pink"]:
            n = ss.lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400], n)
        p = np.mean(a**2) + 1e-12
        n *= np.sqrt(p / (np.mean(n**2) + 1e-12) / 10 ** (c["snr"] / 10))
        a = a + n
    return a.astype(np.float32)


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    enc = MimiEncoder(EncCfg()).eval()
    load_moshi(enc, load_file(f"{W}/kmimi/moshi_tokenizer.safetensors"))
    meter = pyloudnorm.Meter(SR)
    lab = json.load(open(f"{W}/synth/labels.json"))
    items = [(it, l["voice"]) for s in ("shard0", "shard1")
             for it, l in zip(np.load(f"{W}/synth/{s}.npy", allow_pickle=True), lab[s])]
    for k in range(1, K + 1):
        path = f"{W}/cache/synfeat_aug{k}.npy"
        if os.path.exists(path):
            continue
        arr = np.empty(len(items), dtype=object)
        with torch.no_grad():
            for i, (it, v) in enumerate(items):
                a = apply(it["audio"].astype(np.float32) / 32767, channel(zlib.crc32(f"{v}/{k}".encode())))
                a = np.clip(loud(a, -18.0, meter), -1, 1)
                arr[i] = enc.seanet(torch.from_numpy(a)[None, None])[0].numpy().astype(np.float16)
        np.save(path + ".tmp.npy", arr, allow_pickle=True)
        os.replace(path + ".tmp.npy", path)
        print("done variant", k, flush=True)

"""Post-step on generated codec latents: per-channel statistics of the generated frames are moved
towards those of the reference's own latents (codec-encoder stand-in over the whole reference),
then decoded. Verifier-free; uses only the reference.

Usage: latent_match.py IN_DIR OUT_DIR MODE   (MODE: mean | meanstd)
"""
import glob, os, shutil, sys

import numpy as np
import pyloudnorm
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_spk import loud
from train_emain import load_decoder, load_emain
from train_spk import W

if __name__ == "__main__":
    src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    alpha = float(os.environ.get("ALPHA", "1.0"))
    os.makedirs(dst, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    dec, emain = load_decoder(), load_emain(f"{W}/emain.safetensors").eval()
    meter = pyloudnorm.Meter(24000)
    refs = {}
    for ref in glob.glob(f"{src}/*_ref.wav"):
        spk = os.path.basename(ref)[: -len("_ref.wav")]
        a = np.clip(loud(sf.read(ref, dtype="float32")[0], -22.0, meter), -1, 1)
        a = np.pad(a, (0, (-len(a)) % 1920))
        with torch.no_grad():
            z = emain(torch.from_numpy(a)[None, None])[0].T
        # silent frames would drag the statistics: keep the louder 80% of frames
        e = torch.from_numpy(a).reshape(-1, 1920).square().mean(1)
        z = z[e >= e.quantile(0.2)]
        refs[spk] = (z.mean(0), z.std(0))
        shutil.copy(ref, f"{dst}/{spk}_ref.wav")
    for f in sorted(glob.glob(f"{src}/*_clone*.latents.npy")):
        name = os.path.basename(f).replace(".latents.npy", "")
        spk = name.split("_clone")[0]
        n = len(np.load(f"{src}/{spk}_prefix.latents.npy"))
        z = torch.from_numpy(np.load(f))
        g = z[n:]
        mu, sd = refs[spk]
        if mode == "mean":
            g2 = g - g.mean(0) + mu
        else:
            g2 = (g - g.mean(0)) / (g.std(0) + 1e-5) * sd + mu
        z2 = torch.cat([z[:n], g + alpha * (g2 - g)])
        with torch.no_grad():
            wav = dec.decode(z2[None]).audio_values[0, 0].numpy()
        sf.write(f"{dst}/{name}.wav", wav[n * 1920 :], 24000, subtype="FLOAT")
    print("done", dst)

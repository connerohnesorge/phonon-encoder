"""Second round of the codec-encoder stand-in: closer to Phonon's real codec encoder on real speech.

Same two signals as train_emain.py (exact synthetic pairs from Phonon; real speech through the frozen
decoder, multi-resolution mel), with more of both (3,200 synthetic pairs, the ~680-speaker real
pool), the late SEANet stages trainable (TRAIN_FROM onwards), audio encoded online, and longer
training. Starts from emain.safetensors.
"""
import glob, json, os, random, sys, time

import numpy as np
import pyloudnorm
import soundfile as sf
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mimi_torch import SConv
from train_emain import FS, MelLoss, latent_stats, load_decoder, load_emain

W = os.environ.get("PHONON_WORK", "/home/user/work")
CROP = 40
WARM = 6
RCROP = 30
TRAIN_FROM = os.environ.get("TRAIN_FROM", "9")


def seanet_partial(enc, x, split):
    """SEANet with the layers before `split` under no_grad."""
    m, order = enc.mimi.encoder.model, enc.mimi.encoder.order
    with torch.no_grad():
        for i, k in enumerate(order[:split]):
            if isinstance(m[k], SConv) and i > 0:
                x = F.elu(x)
            x = m[k](x)
    for i, k in enumerate(order[split:], start=split):
        if isinstance(m[k], SConv) and i > 0:
            x = F.elu(x)
        x = m[k](x)
    return x


def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    steps = int(os.environ.get("STEPS", "3000"))
    out_path = f"{W}/emain2.safetensors"
    synth = []
    for f in sorted(glob.glob(f"{W}/synth/shard[0-9]*.npy")):
        if "shard99" not in f:
            synth += list(np.load(f, allow_pickle=True))
    held = list(np.load(f"{W}/synth/shard99.npy", allow_pickle=True))
    meta = [json.loads(l) for f in ("meta.jsonl", "meta_more.jsonl") for l in open(f"{W}/audio/{f}")]
    real = [m["path"] for m in meta if m["split"] == "train"]
    rng.shuffle(real)
    real_held, real = real[:24], real[24:]
    print(f"synth {len(synth)} real {len(real)}", flush=True)
    mu, sd = latent_stats()
    meter = pyloudnorm.Meter(24000)

    enc = load_emain(os.environ.get("INIT", f"{W}/emain.safetensors"))
    dec = load_decoder()
    mel = MelLoss()
    order = enc.mimi.encoder.order
    split = order.index(TRAIN_FROM)
    for p in enc.parameters():
        p.requires_grad_(False)
    params = [p for k in order[split:] for p in enc.mimi.encoder.model[k].parameters()]
    params += list(enc.mimi.encoder_transformer.parameters()) + list(enc.mimi.downsample.parameters()) + list(enc.proj.parameters())
    for p in params:
        p.requires_grad_(True)
    lr = float(os.environ.get("LR", "1e-4"))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=float(os.environ.get("WARMUP", "0.03")))

    def encode(x):  # [B, 1, N] -> [B, T, 32]
        return enc.proj(enc.mimi.head(seanet_partial(enc, enc.mimi.pad_input(x), split))).transpose(1, 2)

    def real_crop(path, frames):
        a, _ = sf.read(path, dtype="float32")
        n = frames * FS
        if len(a) > n:
            s = rng.randrange(0, (len(a) - n) // FS + 1) * FS
            a = a[s : s + n]
        a = np.pad(a, (0, n - len(a)))
        a = a - a.mean()
        L = meter.integrated_loudness(a.astype(np.float64))
        if np.isfinite(L):
            a = a * 10 ** ((-22.0 - L) / 20)
        return torch.from_numpy(np.clip(a, -1, 1).astype(np.float32))

    def synth_crop(it):
        a = it["audio"].astype(np.float32) / 32767
        z = torch.from_numpy(it["latents"].astype(np.float32))
        s = rng.randrange(0, max(1, len(z) - CROP + 1))
        z = z[s : s + CROP]
        a = a[s * FS : (s + len(z)) * FS]
        m = torch.ones(CROP)
        if s > 0:
            m[:WARM] = 0
        m[len(z):] = 0
        return torch.from_numpy(np.pad(a, (0, CROP * FS - len(a)))), F.pad(z, (0, 0, 0, CROP - len(z))), m

    def evaluate():
        enc.eval()
        with torch.no_grad():
            num = den = 0.0
            for it in held:
                a = torch.from_numpy(it["audio"].astype(np.float32) / 32767)[None, None]
                z = torch.from_numpy(it["latents"].astype(np.float32))
                p = encode(a)[0][: len(z)]
                num += (((p - z) / sd) ** 2).sum().item()
                den += (((z - z.mean(0)) / sd) ** 2).sum().item()
            r = random.Random(1)
            ml = 0.0
            for p in real_held:
                x = real_crop(p, 60)[None, None]
                y = dec.decode(encode(x)).audio_values[..., : x.shape[-1]]
                ml += mel(y[:, 0], x[:, 0]).item() / len(real_held)
        enc.train()
        return 1 - num / den, ml

    r2, ml = evaluate()
    print(f"init: synth R2 {r2:.3f} real mel {ml:.3f}", flush=True)
    t0 = time.time()
    for step in range(1, steps + 1):
        a, z, m = zip(*(synth_crop(rng.choice(synth)) for _ in range(6)))
        a, z, m = torch.stack(a)[:, None], torch.stack(z), torch.stack(m)
        p = encode(a)
        l_syn = ((((p - z) / sd) ** 2).mean(-1) * m).sum() / m.sum()
        x = torch.stack([real_crop(rng.choice(real), RCROP) for _ in range(3)])[:, None]
        y = dec.decode(encode(x)).audio_values[..., : x.shape[-1]]
        l_mel = mel(y[:, 0], x[:, 0])
        loss = l_syn + l_mel
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0:
            print(f"step {step} syn {l_syn.item():.3f} mel {l_mel.item():.3f} {time.time() - t0:.0f}s", flush=True)
        if step % 250 == 0 or step == steps:
            r2, ml = evaluate()
            print(f"eval {step}: synth R2 {r2:.3f} real mel {ml:.3f}", flush=True)
            save_file({k: v.contiguous() for k, v in enc.state_dict().items()}, out_path)


if __name__ == "__main__":
    main()

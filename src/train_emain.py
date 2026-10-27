"""Trains a stand-in for Phonon's (unreleased) codec encoder: 24 kHz audio -> 32-d latents at 12.5 Hz.

Two signals, both from released Phonon weights only:
- synthetic pairs: latents Phonon generated and the audio its decoder made of them (exact targets);
- real speech: decoder(encoder(x)) must match x (multi-resolution log-mel), decoder frozen.
Built on Kyutai's pretrained Mimi encoder (SEANet front frozen, 25 Hz), whose transformer and
12.5 Hz downsampling are fine-tuned, plus a new 512 -> 32 projection. (Pocket TTS's public
checkpoint ships its codec encoder zeroed, so it cannot serve as the start.)
"""
import glob, json, os, random, sys, time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from safetensors.torch import load_file, save_file
from transformers import PocketTTSAudioTokenizerModel

sys.path.insert(0, os.path.dirname(__file__))
from mimi_torch import EncCfg, MimiEncoder, load_moshi

W = os.environ.get("PHONON_WORK", "/home/user/work")
FS = 1920
CROP = 40  # frames
WARM = 6  # frames without loss at the start of a crop taken mid-utterance
RCROP = 25  # frames of the real-speech crops (2 s)
N_REAL = 1500


class EMain(torch.nn.Module):
    """Kyutai Mimi encoder + a 1x1 projection to the 32-d codec latent: `[B, 1, N]` -> `[B, 32, T]`."""

    def __init__(self):
        super().__init__()
        self.mimi = MimiEncoder(EncCfg())
        self.proj = torch.nn.Conv1d(512, 32, 1)

    @property
    def encoder(self):
        """The SEANet front, kept frozen."""
        return self.mimi.encoder

    def seanet(self, x):
        return self.mimi.seanet(x)

    def head(self, h):
        return self.proj(self.mimi.head(h))

    def forward(self, x):
        return self.head(self.seanet(x))


def load_emain(path=None):
    enc = EMain()
    if path is None:
        load_moshi(enc.mimi, load_file(f"{W}/kmimi/moshi_tokenizer.safetensors"))
        mu, _ = latent_stats()
        torch.nn.init.normal_(enc.proj.weight, std=0.01)
        with torch.no_grad():
            enc.proj.bias.copy_(mu)
    else:
        enc.load_state_dict(load_file(path))
    return enc


def load_decoder():
    dec = PocketTTSAudioTokenizerModel.from_pretrained(f"{W}/pmimi", dtype=torch.float32).eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


class MelLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mels = torch.nn.ModuleList(
            torchaudio.transforms.MelSpectrogram(24000, n_fft=n, hop_length=n // 4, n_mels=m, power=1.0)
            for n, m in ((512, 64), (1024, 80), (2048, 128))
        )

    def forward(self, y, x):
        return sum(F.l1_loss(torch.log(m(y) + 1e-5), torch.log(m(x) + 1e-5)) for m in self.mels) / len(self.mels)


def latent_stats():
    sd = load_file(f"{W}/model/model.safetensors")
    return sd["flow_lm.emb_mean"].float(), sd["flow_lm.emb_std"].float()


def main():
    torch.manual_seed(0)
    random.seed(0)
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    steps = int(os.environ.get("STEPS", "1500"))
    synth = []
    for f in sorted(glob.glob(f"{W}/synth/shard*.npy")):
        if "shard99" in f:
            continue
        synth += list(np.load(f, allow_pickle=True))
    held = list(np.load(f"{W}/synth/shard99.npy", allow_pickle=True))
    meta = [json.loads(l) for l in open(f"{W}/audio/meta.jsonl")]
    real = [m["path"] for m in meta if m["split"] == "train"]
    random.shuffle(real)
    real_held, real = real[:16], real[16:]
    print(f"synth {len(synth)} real {len(real)}", flush=True)
    mu, sd = latent_stats()
    gain = float(os.environ.get("REAL_GAIN_LUFS", "-22"))

    enc = load_emain()
    dec = load_decoder()
    mel = MelLoss()
    import pyloudnorm
    meter = pyloudnorm.Meter(24000)

    def real_crop(path, frames=CROP):
        a, _ = sf.read(path, dtype="float32")
        n = frames * FS
        if len(a) > n:
            s = random.randrange(0, (len(a) - n) // FS + 1) * FS
            a = a[s : s + n]
        a = np.pad(a, (0, n - len(a)))
        a = a - a.mean()
        L = meter.integrated_loudness(a.astype(np.float64))
        if np.isfinite(L):
            a = a * 10 ** ((gain - L) / 20)
        return torch.from_numpy(np.clip(a, -1, 1).astype(np.float32))

    # The SEANet front stays frozen: its features are computed once, and only the encoder's
    # transformer and downsampling conv train.
    for p_ in enc.encoder.parameters():
        p_.requires_grad_(False)
    params = [p_ for p_ in enc.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.05)
    hop = 2  # SEANet frames per codec frame
    t0 = time.time()
    with torch.no_grad():
        sfeat = [enc.seanet(torch.from_numpy(it["audio"].astype(np.float32) / 32767)[None, None])[0].half() for it in synth]
        rpool = []
        for path in real[:N_REAL]:
            x = real_crop(path, RCROP)
            rpool.append((enc.seanet(x[None, None])[0].half(), x))
    print(f"cached features in {time.time() - t0:.0f}s", flush=True)

    def synth_crop(i):
        it, h = synth[i], sfeat[i]
        z = torch.from_numpy(it["latents"].astype(np.float32))
        T = len(z)
        s = random.randrange(0, max(1, T - CROP + 1))
        z = z[s : s + CROP]
        h = h[:, s * hop : (s + len(z)) * hop].float()
        m = torch.ones(CROP)
        if s > 0:
            m[:WARM] = 0
        m[len(z):] = 0
        z = F.pad(z, (0, 0, 0, CROP - len(z)))
        h = F.pad(h, (0, CROP * hop - h.shape[1]))
        return h, z, m

    def evaluate():
        enc.eval()
        with torch.no_grad():
            num = den = 0.0
            for it in held:
                a = torch.from_numpy(it["audio"].astype(np.float32) / 32767)[None, None]
                z = torch.from_numpy(it["latents"].astype(np.float32))
                p = enc(a)[0].T[: len(z)]
                num += (((p - z) / sd) ** 2).sum().item()
                den += (((z - z.mean(0)) / sd) ** 2).sum().item()
            ml = 0.0
            for p in real_held:
                x = real_crop(p)[None, None]
                y = dec.decode(enc(x).transpose(1, 2)).audio_values[..., : x.shape[-1]]
                ml += mel(y[:, 0], x[:, 0]).item() / len(real_held)
        enc.train()
        return 1 - num / den, ml

    t0 = time.time()
    for step in range(1, steps + 1):
        h, z, m = zip(*(synth_crop(random.randrange(len(synth))) for _ in range(8)))
        h, z, m = torch.stack(h), torch.stack(z), torch.stack(m)
        p = enc.head(h).transpose(1, 2)
        l_syn = ((((p - z) / sd) ** 2).mean(-1) * m).sum() / m.sum()
        hr, x = zip(*(random.choice(rpool) for _ in range(3)))
        hr, x = torch.stack(hr).float(), torch.stack(x)[:, None]
        y = dec.decode(enc.head(hr).transpose(1, 2)).audio_values[..., : x.shape[-1]]
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
            save_file({k: v.contiguous() for k, v in enc.state_dict().items()}, f"{W}/emain.safetensors")


if __name__ == "__main__":
    main()

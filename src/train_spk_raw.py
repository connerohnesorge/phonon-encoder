"""Speaker-encoder training from raw audio, with the SEANet tail unfrozen.

Same objective as train_spk.py phases 2-3 (half real speech, flow LM's own loss; half Phonon speech
in catalog voices, distilled towards the frozen LM's behaviour under the voice's released
`speaker_wavs`), with three differences:
- the encoder runs on raw audio each step, so its late SEANet stages (TRAIN_FROM onwards: the
  200 Hz and 25 Hz convolutions) train too; the early, high-rate layers stay frozen;
- every reference goes through refcond.prepare_reference, exactly what the runtime does;
- synthetic references get a fresh random recording channel (prep_synaug.channel) per example.
"""
import glob, json, os, random, sys, time
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_synaug import apply, channel
from refcond import prepare_reference
from train_spk import (MAX_TGT, W, catalog_stats, flow_loss, hidden, load_cache, load_speaker_encoder,
                       load_tts)

TRAIN_FROM = int(os.environ.get("TRAIN_FROM", "9"))  # first trainable SEANet `model.N`
REF_RAW_S = 16.0  # raw audio gathered per reference before conditioning (it keeps <= 10 s)


def chat(proc, exs, prompt_lens=None, prompts=None):
    convs = [[{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": t}]},
              {"role": "assistant", "content": [{"type": "audio"}]}] for _, t in exs]
    if prompts is None:
        prompts = [torch.zeros(n, 512) for n in prompt_lens]
    return proc(text=proc.apply_chat_template(convs, tokenize=False), audio_prompt_latents=prompts,
                labels=[torch.from_numpy(z.astype(np.float32)) for z, _ in exs], return_tensors="pt")


class Encoder:
    """The speaker encoder on raw audio: frozen early SEANet under no_grad, the rest with grads."""

    def __init__(self, enc):
        self.enc = enc
        self.order = enc.encoder.order
        # TRAIN_FROM < 0 keeps the whole SEANet frozen (only the transformer and downsampler train)
        self.split = len(self.order) if TRAIN_FROM < 0 else self.order.index(str(TRAIN_FROM))

    def trainable(self):
        ps = [p for k in self.order[self.split:] for p in self.enc.encoder.model[k].parameters()]
        return ps + list(self.enc.encoder_transformer.parameters()) + list(self.enc.downsample.parameters())

    def __call__(self, wavs):
        """List of 1-D float arrays -> list of [frames, 512] prompt latents."""
        import torch.nn.functional as F
        from mimi_torch import SConv

        outs = []
        for w in wavs:
            x = self.enc.pad_input(torch.from_numpy(w)[None, None])
            m = self.enc.encoder.model
            with torch.no_grad():
                for i, k in enumerate(self.order[: self.split]):
                    if isinstance(m[k], SConv) and i > 0:
                        x = F.elu(x)
                    x = m[k](x)
            for i, k in enumerate(self.order[self.split:], start=self.split):
                if isinstance(m[k], SConv) and i > 0:
                    x = F.elu(x)
                x = m[k](x)
            outs.append(self.enc.head(x)[0].T)
        return outs


def pad(lats):
    lat = torch.nn.utils.rnn.pad_sequence(lats, batch_first=True)
    mask = torch.nn.utils.rnn.pad_sequence([torch.ones(len(l), dtype=torch.long) for l in lats], batch_first=True)
    return lat, mask


def read(path):
    return sf.read(path, dtype="float32")[0]


def main():
    seed = int(os.environ.get("SEED", "0"))
    steps = int(os.environ.get("STEPS", "1500"))
    bs = int(os.environ.get("BS", "8"))
    lr = float(os.environ.get("LR", "5e-5"))
    p_aug = float(os.environ.get("P_AUG", "0.75"))
    out_path = os.environ.get("OUT", f"{W}/spk_enc_raw.safetensors")
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    rng = random.Random(seed)

    # real speakers: every training utterance with a codec-latent target
    zs = load_cache("z")
    meta = [json.loads(l) for f in ("meta.jsonl", "meta_more.jsonl") if os.path.exists(f"{W}/audio/{f}")
            for l in open(f"{W}/audio/{f}")]
    by_spk = defaultdict(list)
    for m in meta:
        if m["split"] == "train" and m["path"] in zs:
            by_spk[m["spk"]].append(m)
    spks = sorted(s for s, us in by_spk.items() if len(us) >= 4)
    random.Random(123).shuffle(spks)
    val_spk, train_spk = spks[:12], spks[12:]

    # catalog voices: Phonon's own speech in each, labelled
    lab = json.load(open(f"{W}/synth/labels.json"))
    by_voice = defaultdict(list)
    for s in sorted(glob.glob(f"{W}/synth/shard[0-9]*.npy")):
        name = os.path.basename(s)[:-4]
        if name == "shard99":
            continue
        for i, it in enumerate(np.load(s, allow_pickle=True)):
            v, t = (it["voice"], it["text"]) if "voice" in it else (lab[name][i]["voice"], lab[name][i]["text"])
            by_voice[v].append((it["audio"], it["latents"], t))
    catalog = {os.path.basename(p)[:-12]: load_file(p)["speaker_wavs"][0].T.float()
               for p in sorted(glob.glob(f"{W}/voices/voices/*.safetensors"))}
    held = set(sorted(catalog)[:: len(catalog) // 10][:10])
    train_voices = [v for v in by_voice if v not in held and len(by_voice[v]) >= 3]
    held_voices = [v for v in sorted(by_voice) if v in held and len(by_voice[v]) >= 3]
    print(f"real speakers: train {len(train_spk)} val {len(val_spk)} | voices: train {len(train_voices)} "
          f"held-out {len(held_voices)} | synthetic utterances {sum(len(x) for x in by_voice.values())}", flush=True)

    def real_example(r, spk=None):
        us = by_spk[spk or r.choice(train_spk)][:]
        r.shuffle(us)
        tgt = next(u for u in us if len(zs[u["path"]]) <= MAX_TGT)
        parts, n = [], 0.0
        for u in us:
            if u is not tgt:
                parts.append(read(u["path"]))
                n += len(parts[-1]) / 24000
                if n >= REF_RAW_S:
                    break
        return prepare_reference(np.concatenate(parts)), (zs[tgt["path"]], tgt["text"])

    def synth_example(r, v=None, aug=None, other=None):
        v = v or r.choice(train_voices)
        its = by_voice[v][:]
        r.shuffle(its)
        tgt = next(i for i in its if len(i[1]) <= MAX_TGT)
        src = by_voice[other] if other else its
        parts, n = [], 0.0
        for i in src:
            if i is not tgt:
                parts.append(i[0].astype(np.float32) / 32767)
                n += len(parts[-1]) / 24000
                if n >= REF_RAW_S:
                    break
        a = np.concatenate(parts)
        if aug if aug is not None else r.random() < p_aug:
            a = apply(a, channel(r.randrange(1 << 31)))
        return prepare_reference(a), (tgt[1], tgt[2]), v

    proc, model = load_tts()
    enc = load_speaker_encoder(os.environ.get("INIT"))
    E = Encoder(enc)
    for p in enc.parameters():
        p.requires_grad_(False)
    params = E.trainable()
    for p in params:
        p.requires_grad_(True)
    print(f"trainable parameters {sum(p.numel() for p in params) / 1e6:.1f}M "
          f"({'SEANet frozen' if TRAIN_FROM < 0 else f'SEANet from model.{TRAIN_FROM}'})", flush=True)
    mu_c, sd_c = catalog_stats()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.03)

    # fixed validation: real val speakers (own vs another speaker's reference) and held-out
    # voices with augmented references (own vs another voice's audio)
    vr = random.Random(7)
    val_real = [real_example(vr, s) for s in val_spk for _ in range(3)]
    val_real_wrong = [(val_real[(i + 3) % len(val_real)][0], e[1]) for i, e in enumerate(val_real)]
    vs = random.Random(11)
    val_syn = [synth_example(vs, v, aug=True) for v in held_voices for _ in range(2)]
    val_syn_wrong = [synth_example(random.Random(11 + i), v, aug=True, other=held_voices[(k + 1) % len(held_voices)])
                     for i, (k, v) in enumerate((k, v) for k, v in enumerate(held_voices) for _ in range(2))]

    def real_loss(exs, n_noise=4, gen=None):
        lats = E([w for w, _ in exs])
        lat, mask = pad(lats)
        inputs = chat(proc, [e for _, e in exs], [len(l) for l in lats])
        lf, le = flow_loss(model, inputs, lat, mask, n_noise=n_noise, gen=gen)
        return lf + 0.1 * le, lat[mask.bool()]

    def syn_loss(exs):
        tgts = [e for _, e, _ in exs]
        t_in = chat(proc, tgts, prompts=[catalog[v] for _, _, v in exs])
        with torch.no_grad():
            h_t = hidden(model, t_in, t_in["audio_prompt_latents"], t_in["audio_prompt_latents_mask"])
        lats = E([w for w, _, _ in exs])
        lat, mask = pad(lats)
        s_in = chat(proc, tgts, [len(l) for l in lats])
        h_s = hidden(model, s_in, lat, mask)
        lf, le = flow_loss(model, s_in, lat, mask)
        return (h_s - h_t).square().mean(), lf + 0.1 * le, lat[mask.bool()]

    def validate(step):
        enc.eval()
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            r_own = np.mean([real_loss(val_real[i:i + 6], 8, g)[0].item() for i in range(0, len(val_real), 6)])
            r_wrong = np.mean([real_loss(val_real_wrong[i:i + 6], 8, g)[0].item() for i in range(0, len(val_real), 6)])
            s_own = np.mean([syn_loss(val_syn[i:i + 5])[0].item() for i in range(0, len(val_syn), 5)])
            s_wrong = np.mean([syn_loss(val_syn_wrong[i:i + 5])[0].item() for i in range(0, len(val_syn), 5)])
        enc.train()
        print(f"val {step}: real own {r_own:.4f} wrong {r_wrong:.4f} | held-out voices (augmented refs) "
              f"distill own {s_own:.4f} wrong {s_wrong:.4f}", flush=True)

    state_path, start = out_path + ".state.pt", 1
    if os.path.exists(state_path):
        st = torch.load(state_path, weights_only=False)
        enc.load_state_dict(st["enc"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        rng.setstate(st["rng"])
        torch.set_rng_state(st["torch_rng"])
        start = st["step"] + 1
        print(f"resumed at step {start}", flush=True)
    elif not os.environ.get("DRY"):
        validate(0)
    t0 = time.time()
    for step in range(start, steps + 1):
        lr_real, flat_r = real_loss([real_example(rng) for _ in range(bs // 2)])
        l_dist, lr_syn, flat_s = syn_loss([synth_example(rng) for _ in range(bs - bs // 2)])
        flat = torch.cat([flat_r, flat_s])
        l_mom = (((flat.mean(0) - mu_c) / sd_c) ** 2).mean() + ((flat.std(0).log() - sd_c.log()) ** 2).mean()
        loss = (lr_real + lr_syn) / 2 + 0.05 * l_mom + l_dist
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if os.environ.get("DRY"):
            print(f"dry step ok: real {lr_real.item():.4f} syn {lr_syn.item():.4f} dist {l_dist.item():.4f} "
                  f"{time.time() - t0:.1f}s", flush=True)
            return
        if step % 25 == 0:
            print(f"step {step} real {lr_real.item():.4f} syn {lr_syn.item():.4f} dist {l_dist.item():.4f} "
                  f"mom {l_mom.item():.3f} {time.time() - t0:.0f}s", flush=True)
        if step % 250 == 0 or step == steps:
            validate(step)
            save_file({k: v.contiguous() for k, v in enc.state_dict().items()}, out_path)
            st = {"enc": enc.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                  "rng": rng.getstate(), "torch_rng": torch.get_rng_state(), "step": step}
            torch.save(st, state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)


if __name__ == "__main__":
    main()

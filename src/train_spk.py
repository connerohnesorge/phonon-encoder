"""Trains the speaker encoder (Phonon's `mimi_speaker`) through the frozen Phonon flow LM.

The encoder maps a reference recording of a speaker to the voice-prompt latents the flow LM is
conditioned on. Its only training signal is the flow LM's own objective: with the prompt made from
reference audio A of a speaker, the (frozen) model must assign high likelihood (flow-matching loss,
plus the end-of-speech loss) to the codec latents of a different utterance B of the same speaker,
given B's text. No speaker-verification model is used.

The SEANet front of the encoder stays frozen (features cached by prep_spk.py); its transformer and
downsampling conv are trained, initialised from Kyutai's Mimi encoder, the speaker encoder's
architecture in Phonon's config.
"""
import glob, json, os, random, sys, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration

sys.path.insert(0, os.path.dirname(__file__))
from mimi_torch import EncCfg, MimiEncoder, load_moshi

W = os.environ.get("PHONON_WORK", "/home/user/work")
REV = "92741c5b07d085ff840e26fea8ce43110b87852f"
REF_FRAMES = 250  # SEANet frames (25 Hz): 10 s, the runtime's reference cap
MAX_TGT = 125  # codec frames (12.5 Hz): 10 s
VAL_SPK = 12


def load_cache(what):
    out = {}
    for f in sorted(glob.glob(f"{W}/cache/{what}_[0-9]*.npy")):
        out.update(np.load(f, allow_pickle=True).item())
    return out


def load_tts():
    proc = AutoProcessor.from_pretrained("gradium/phonon-31m-en", revision=REV)
    model = GradiumPhononTTSForConditionalGeneration.from_pretrained(
        "gradium/phonon-31m-en", revision=REV, dtype=torch.float32
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return proc, model


def load_speaker_encoder(path=None):
    enc = MimiEncoder(EncCfg())
    load_moshi(enc, load_file(f"{W}/kmimi/moshi_tokenizer.safetensors"))
    if path:
        enc.load_state_dict(load_file(path), strict=False)
    return enc


class Batcher:
    def __init__(self, proc, meta, feats, zs, speakers, rng):
        self.proc, self.feats, self.zs, self.rng = proc, feats, zs, rng
        self.by_spk = defaultdict(list)
        for m in meta:
            if m["spk"] in speakers and m["path"] in feats and m["path"] in zs:
                self.by_spk[m["spk"]].append(m)
        self.speakers = [s for s in speakers if len(self.by_spk[s]) >= 4]

    def example(self, spk=None, full_ref=False):
        rng = self.rng
        spk = spk or rng.choice(self.speakers)
        utts = self.by_spk[spk][:]
        rng.shuffle(utts)
        tgt = next(u for u in utts if len(self.zs[u["path"]]) <= MAX_TGT)
        n_ref = REF_FRAMES if full_ref or rng.random() < 0.7 else 2 * rng.randint(63, REF_FRAMES // 2)
        parts, n = [], 0
        for u in utts:
            if u is tgt:
                continue
            parts.append(self.feats[u["path"]])
            n += parts[-1].shape[1]
            if n >= n_ref:
                break
        ref = np.concatenate(parts, 1)[:, :n_ref]
        ref = ref[:, : ref.shape[1] // 2 * 2]
        return ref, self.zs[tgt["path"]], tgt["text"]

    def collate(self, exs):
        refs = [torch.from_numpy(r.astype(np.float32)) for r, _, _ in exs]
        n_prompt = [r.shape[1] // 2 for r in refs]  # stride-2 downsample
        convs = [
            [
                {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": t}]},
                {"role": "assistant", "content": [{"type": "audio"}]},
            ]
            for _, _, t in exs
        ]
        text = self.proc.apply_chat_template(convs, tokenize=False)
        inputs = self.proc(
            text=text,
            audio_prompt_latents=[torch.zeros(n, 512) for n in n_prompt],
            labels=[torch.from_numpy(z.astype(np.float32)) for _, z, _ in exs],
            return_tensors="pt",
        )
        return refs, n_prompt, inputs


def encode_refs(enc, refs, n_prompt):
    """SEANet features -> prompt latents, padded `[B, max, 512]` with mask."""
    outs = [enc.head(r[None])[0].T for r in refs]
    for o, n in zip(outs, n_prompt):
        assert o.shape[0] == n, (o.shape, n)
    lat = torch.nn.utils.rnn.pad_sequence(outs, batch_first=True)
    mask = torch.nn.utils.rnn.pad_sequence([torch.ones(n, dtype=torch.long) for n in n_prompt], batch_first=True)
    return lat, mask


def flow_loss(model, inputs, lat, mask, n_noise=4, gen=None):
    """Flow-matching (LSD diagonal) loss of the target frames + 0.1 x end-of-speech loss."""
    cfg = model.config
    ids, am = inputs["input_ids"], inputs["attention_mask"]
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    labels, lmask = inputs["labels"], inputs["labels_mask"]
    normed = (labels - model.latent_mean) / model.latent_std
    out = model.model(
        input_ids=ids,
        attention_mask=am,
        position_ids=pos,
        audio_prompt_latents=lat,
        audio_prompt_latents_mask=mask,
        audio_latents=normed,
        audio_latents_mask=lmask,
    )
    h = out.last_hidden_state
    latent_pos = (ids == cfg.audio_bos_token_id) | (ids == cfg.audio_latent_token_id)
    frame_mask = torch.cat([torch.ones_like(lmask[:, :1]), lmask], 1).bool()
    is_target = torch.cat([lmask, torch.zeros_like(lmask[:, :1])], 1)[frame_mask].bool()
    hl = h[latent_pos]
    cond = hl[is_target]
    tgt = normed[lmask.bool()]
    cond = cond.repeat(n_noise, 1)
    tgt = tgt.repeat(n_noise, 1)
    noise = torch.randn(tgt.shape, generator=gen)
    t = torch.sigmoid(torch.randn(tgt.shape[0], 1, generator=gen) * cfg.lsd_time_logit_std + cfg.lsd_time_logit_mean)
    x_t = t * tgt + (1 - t) * noise
    v = model.flow_head(x_t, cond, times=[t, t])
    l_flow = (v - (tgt - noise)).square().mean()
    eos_logits = model.eos_head(hl)[:, 0]
    l_eos = torch.where(~is_target, F.softplus(-eos_logits), F.softplus(eos_logits)).mean()
    return l_flow, l_eos


class SynthBatcher:
    """Phonon's own utterances in released catalog voices: the speaker encoder hears other
    utterances of the voice; the teacher is the frozen LM prompted with the voice's released
    `speaker_wavs`."""

    def __init__(self, proc, voices, rng, keep):
        lab = json.load(open(f"{W}/synth/labels.json"))
        # variant 0 is the clean audio; variants 1..K went through simulated recording channels
        self.feats = [np.load(f"{W}/cache/synfeat.npy", allow_pickle=True)]
        self.feats += [np.load(f"{W}/cache/synfeat_aug{k}.npy", allow_pickle=True)
                       for k in range(1, int(os.environ.get("AUG", "0")) + 1)]
        self.p_aug = float(os.environ.get("P_AUG", "0.75"))
        items, k = [], 0
        for s in ("shard0", "shard1"):
            for it, l in zip(np.load(f"{W}/synth/{s}.npy", allow_pickle=True), lab[s]):
                items.append((l["voice"], it["latents"], l["text"], k))
                k += 1
        self.by_voice = defaultdict(list)
        for it in items:
            if keep(it[0]):
                self.by_voice[it[0]].append(it)
        self.voices = [v for v, its in self.by_voice.items() if len(its) >= 3]
        self.proc, self.rng, self.catalog = proc, rng, voices

    def example(self, v=None):
        rng = self.rng
        v = v or rng.choice(self.voices)
        its = self.by_voice[v][:]
        rng.shuffle(its)
        tgt = next(i for i in its if len(i[1]) <= MAX_TGT)
        var = 0
        if len(self.feats) > 1 and rng.random() < self.p_aug:
            var = rng.randrange(1, len(self.feats))
        parts, n = [], 0
        for i in its:
            if i is tgt:
                continue
            parts.append(self.feats[var][i[3]])
            n += parts[-1].shape[1]
            if n >= REF_FRAMES:
                break
        ref = np.concatenate(parts, 1)[:, :REF_FRAMES]
        ref = ref[:, : ref.shape[1] // 2 * 2]
        return ref, tgt[1], tgt[2], v

    def teacher_inputs(self, exs):
        convs = [
            [
                {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": t}]},
                {"role": "assistant", "content": [{"type": "audio"}]},
            ]
            for _, _, t, _ in exs
        ]
        return self.proc(
            text=self.proc.apply_chat_template(convs, tokenize=False),
            audio_prompt_latents=[self.catalog[v] for _, _, _, v in exs],
            labels=[torch.from_numpy(z.astype(np.float32)) for _, z, _, _ in exs],
            return_tensors="pt",
        )


def hidden(model, inputs, lat, mask):
    """Backbone hidden states at every text and audio position (the prompt excluded), in order."""
    ids, am = inputs["input_ids"], inputs["attention_mask"]
    normed = (inputs["labels"] - model.latent_mean) / model.latent_std
    h = model.model(
        input_ids=ids, attention_mask=am, position_ids=(am.cumsum(-1) - 1).clamp(min=0),
        audio_prompt_latents=lat, audio_prompt_latents_mask=mask,
        audio_latents=normed, audio_latents_mask=inputs["labels_mask"],
    ).last_hidden_state
    keep = am.bool() & (ids != model.config.audio_prompt_token_id)
    return h[keep]


def distill_loss(model, enc, sb, exs):
    """Student (prompt from the speaker encoder) vs teacher (the released voice) hidden states."""
    refs = [torch.from_numpy(r.astype(np.float32)) for r, _, _, _ in exs]
    n_prompt = [r.shape[1] // 2 for r in refs]
    _, _, s_inputs = Batcher.collate(sb, [(r, z, t) for r, z, t, _ in exs])
    t_inputs = sb.teacher_inputs(exs)
    with torch.no_grad():
        h_t = hidden(model, t_inputs, t_inputs["audio_prompt_latents"], t_inputs["audio_prompt_latents_mask"])
    lat, mask = encode_refs(enc, refs, n_prompt)
    h_s = hidden(model, s_inputs, lat, mask)
    return (h_s - h_t).square().mean(), (s_inputs, lat, mask)


def catalog_stats():
    lat = torch.cat(
        [load_file(p)["speaker_wavs"][0].T for p in sorted(glob.glob(f"{W}/voices/voices/*.safetensors"))]
    ).float()
    return lat.mean(0), lat.std(0)


def main():
    seed = int(os.environ.get("SEED", "0"))
    steps = int(os.environ.get("STEPS", "3000"))
    bs = int(os.environ.get("BS", "8"))
    lr = float(os.environ.get("LR", "1e-4"))
    out_path = os.environ.get("OUT", f"{W}/spk_enc.safetensors")
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    rng = random.Random(seed)

    meta = [json.loads(l) for l in open(f"{W}/audio/meta.jsonl")]
    meta = [m for m in meta if m["split"] == "train"]
    feats, zs = load_cache("feat"), load_cache("z")
    spks = sorted({m["spk"] for m in meta})
    random.Random(123).shuffle(spks)
    val_spk, train_spk = spks[:VAL_SPK], spks[VAL_SPK:]
    proc, model = load_tts()
    tr = Batcher(proc, meta, feats, zs, train_spk, rng)
    va = Batcher(proc, meta, feats, zs, val_spk, random.Random(7))
    print(f"train speakers {len(tr.speakers)} val speakers {len(va.speakers)}", flush=True)
    # fixed validation set: for each val speaker, 3 (ref, target) pairs; also the same targets
    # with the reference of another val speaker (the "wrong voice" control)
    val = [va.example(s, full_ref=True) for s in va.speakers for _ in range(3)]
    wrong = [(val[(i + 3) % len(val)][0], z, t) for i, (_, z, t) in enumerate(val)]

    # phase 2 (SYN=1): half of each batch distils the released voices, from INIT's weights
    syn = bool(int(os.environ.get("SYN", "0")))
    if syn:
        catalog = {os.path.basename(p)[:-12]: load_file(p)["speaker_wavs"][0].T.float()
                   for p in sorted(glob.glob(f"{W}/voices/voices/*.safetensors"))}
        held_voices = set(sorted(catalog)[:: len(catalog) // 10][:10])
        sb = SynthBatcher(proc, catalog, rng, lambda v: v not in held_voices)
        sv = SynthBatcher(proc, catalog, random.Random(11), lambda v: v in held_voices)
        syn_val = [sv.example(v) for v in sv.voices for _ in range(2)]
        syn_wrong = [(syn_val[(i + 2) % len(syn_val)][0],) + e[1:] for i, e in enumerate(syn_val)]
        print(f"synthetic voices: train {len(sb.voices)} held-out {len(sv.voices)}", flush=True)

    enc = load_speaker_encoder(os.environ.get("INIT"))
    for p in enc.encoder.parameters():
        p.requires_grad_(False)
    params = [p for p in enc.parameters() if p.requires_grad]
    mu_c, sd_c = catalog_stats()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.03)

    def evaluate(pairs):
        enc.eval()
        g = torch.Generator().manual_seed(0)
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(pairs), 6):
                refs, n_prompt, inputs = va.collate(pairs[i : i + 6])
                lat, mask = encode_refs(enc, refs, n_prompt)
                lf, le = flow_loss(model, inputs, lat, mask, n_noise=8, gen=g)
                tot += (lf + 0.1 * le).item() * len(refs)
        enc.train()
        return tot / len(pairs)

    def catalog_baseline():
        # the flow LM's loss on the val targets with a released catalog voice as the prompt
        g = torch.Generator().manual_seed(0)
        voices = [load_file(p)["speaker_wavs"][0].T.float() for p in sorted(glob.glob(f"{W}/voices/voices/*.safetensors"))]
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(val), 6):
                refs, n_prompt, inputs = va.collate(val[i : i + 6])
                vs = [voices[(i + j) % len(voices)] for j in range(len(refs))]
                inputs = va.proc(
                    text=va.proc.apply_chat_template(
                        [[{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": t}]},
                          {"role": "assistant", "content": [{"type": "audio"}]}] for _, _, t in val[i : i + 6]],
                        tokenize=False),
                    audio_prompt_latents=vs, labels=[torch.from_numpy(z.astype(np.float32)) for _, z, _ in val[i : i + 6]],
                    return_tensors="pt")
                lf, le = flow_loss(model, inputs, inputs["audio_prompt_latents"], inputs["audio_prompt_latents_mask"], 8, g)
                tot += (lf + 0.1 * le).item() * len(refs)
        return tot / len(val)

    def evaluate_syn(exs):
        enc.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(exs), 5):
                tot += distill_loss(model, enc, sv, exs[i : i + 5])[0].item() * len(exs[i : i + 5])
        enc.train()
        return tot / len(exs)

    def validate(step):
        msg = f"val {step}: own {evaluate(val):.4f} wrong {evaluate(wrong):.4f}"
        if syn:
            msg += f" | held-out voices distill: own {evaluate_syn(syn_val):.4f} wrong {evaluate_syn(syn_wrong):.4f}"
        print(msg, flush=True)

    # resume after a restart: weights, optimizer, schedule and step from the last checkpoint
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
    else:
        if not syn:
            print(f"val loss: catalog voice {catalog_baseline():.4f}", flush=True)
        validate(0)
    t0 = time.time()
    w_dist = float(os.environ.get("W_DIST", "1.0"))
    for step in range(start, steps + 1):
        n_real = bs // 2 if syn else bs
        refs, n_prompt, inputs = tr.collate([tr.example() for _ in range(n_real)])
        lat, mask = encode_refs(enc, refs, n_prompt)
        lf, le = flow_loss(model, inputs, lat, mask)
        flat = lat[mask.bool()]
        l_dist = torch.zeros(())
        if syn:
            l_dist, (s_inputs, s_lat, s_mask) = distill_loss(model, enc, sb, [sb.example() for _ in range(bs - n_real)])
            lf_s, le_s = flow_loss(model, s_inputs, s_lat, s_mask)
            lf, le = (lf + lf_s) / 2, (le + le_s) / 2
            flat = torch.cat([flat, s_lat[s_mask.bool()]])
        l_mom = (((flat.mean(0) - mu_c) / sd_c) ** 2).mean() + ((flat.std(0).log() - sd_c.log()) ** 2).mean()
        loss = lf + 0.1 * le + 0.05 * l_mom + w_dist * l_dist
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0:
            print(f"step {step} flow {lf.item():.4f} eos {le.item():.4f} mom {l_mom.item():.3f} "
                  f"dist {l_dist.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step % 250 == 0 or step == steps:
            validate(step)
            sd = {k: v.contiguous() for k, v in enc.state_dict().items() if not k.startswith("encoder.")}
            save_file(sd, out_path)
            st = {"enc": enc.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                  "rng": rng.getstate(), "torch_rng": torch.get_rng_state(), "step": step}
            torch.save(st, state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)


if __name__ == "__main__":
    main()

"""Diagnostic: speaker guidance on teacher-forced predictions. h = h_null + w (h_voice - h_null),
h_null from the same text and audio history with no voice prompt (as the runtime's CFG null
branch). ECAPA ratio per w for held-out utterances.

Usage: diag_cfg.py VOICE_DIR SPK [SPK ...]
"""
import itertools, os, sys

import librosa
import numpy as np
import pyloudnorm
import soundfile as sf
import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapt_voice import MODELS, SR
from continue_clone import TEMP
from eval_clone import speaker_sets
from prep_spk import loud
from train_emain import load_decoder, load_emain
from train_spk import W
from verifiers import Verifiers, load16

WS = [float(w) for w in os.environ.get("WS", "1,1.5,2,3").split(",")]


@torch.no_grad()
def hidden(model, proc, prompt, text, zn):
    content = ([{"type": "audio"}] if prompt is not None else []) + [{"type": "text", "text": text}]
    kw = {"processor_kwargs": {"audio_prompt_latents": [prompt]}} if prompt is not None else {}
    inp = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True, return_dict=True,
                                   add_generation_prompt=True, **kw)
    n = len(zn)
    ids = torch.cat([inp["input_ids"], torch.full((1, n), model.config.audio_latent_token_id)], 1)
    extra = {k: inp[k] for k in ("audio_prompt_latents", "audio_prompt_latents_mask") if k in inp}
    out = model(input_ids=ids, audio_latents=zn[None], **extra)
    return out.last_hidden_state[0, -n - 1 : -1]


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    voice_dir = sys.argv[1]
    repo, rev = MODELS[os.environ.get("MODEL", "40m")]
    proc = AutoProcessor.from_pretrained(repo, revision=rev)
    model = GradiumPhononTTSForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=torch.float32).eval()
    dec = load_decoder()
    emain = load_emain(os.environ.get("EMAIN", f"{W}/emain.safetensors")).eval()
    V, meter = Verifiers(), pyloudnorm.Meter(SR)
    sets = speaker_sets()
    for s in sys.argv[2:]:
        _, held = sets[s]
        prompt = load_file(f"{voice_dir}/{s}.safetensors")["speaker_wavs"][0].T.float()
        real = [V.embed(load16(u["path"]))["ecapa"] for u in held]
        ceil = np.mean([a @ b for a, b in itertools.combinations(real, 2)])
        res = {w: [] for w in WS}
        for i, u in enumerate(held[:3]):
            a = np.clip(loud(sf.read(u["path"], dtype="float32")[0], -22.0, meter), -1, 1)
            a = np.pad(a, (0, (-len(a)) % 1920))
            with torch.no_grad():
                z = emain(torch.from_numpy(a)[None, None])[0].T
                zn = (z - model.latent_mean) / model.latent_std
                hc, hn = hidden(model, proc, prompt, u["text"], zn), hidden(model, proc, None, u["text"], zn)
                for w in WS:
                    noise = torch.randn(len(z), model.config.latent_dim, generator=torch.Generator().manual_seed(i)) * TEMP**0.5
                    zz = model.denormalize_latents(model.sample_latent(hn + w * (hc - hn), noise, 1))
                    y = dec.decode(zz[None]).audio_values[0, 0].numpy()
                    e = V.embed(librosa.resample(y, orig_sr=SR, target_sr=16000))["ecapa"]
                    res[w] += [float(e @ real[j]) for j in range(len(real)) if j != i]
        print(s, {w: round(np.mean(v) / ceil, 3) for w, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()

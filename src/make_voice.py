"""Raw reference recording -> a cloned Phonon voice file for the Rust runtime.

1. speaker encoder -> voice prompt (`speaker_wavs`), adapted to the reference (adapt_voice.py);
2. the reference's last phrases -> `prefix_latents` + `prefix_text` metadata (continue_clone.py),
   which the runtime speaks every chunk after, as a continuation of the recording.

    make_voice.py 31m|40m OUT_DIR REF.wav [REF.wav ...]      # writes OUT_DIR/<name>.safetensors
    ptts --model <checkpoint> --voice OUT_DIR/<name>.safetensors "Text" -o out.wav

Only the reference recording is used; no speaker-verification model is involved.
"""
import os, sys, time

import librosa
import numpy as np
import pyloudnorm
import soundfile as sf
import torch
from safetensors.torch import save_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration, pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# the adaptation settings that scored best (adapt_voice reads them at import)
for k, v in (("MODE", "full"), ("STEPS", "300"), ("ADAPT_LR", "0.01"), ("L2", "0.1"), ("LM_PRIOR", "0.3"),
             ("LM_PRIOR_STEPS", "300")):
    os.environ.setdefault(k, v)
from adapt_voice import MODELS, SR, adapt
from continue_clone import prefix_from_reference
from train_emain import load_emain
from train_spk import W, load_speaker_encoder


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    repo, rev = MODELS[sys.argv[1]]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    proc = AutoProcessor.from_pretrained(repo, revision=rev)
    model = GradiumPhononTTSForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    enc = load_speaker_encoder(os.environ.get("ENC", f"{W}/spk_enc_final.safetensors")).eval()
    emain = load_emain(os.environ.get("EMAIN", f"{W}/emain.safetensors")).eval()
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
    meter = pyloudnorm.Meter(SR)
    for ref in sys.argv[3:]:
        t = time.time()
        a, sr = sf.read(ref, dtype="float32")
        a = a.mean(1) if a.ndim > 1 else a
        a = librosa.resample(a, orig_sr=sr, target_sr=SR) if sr != SR else a
        name = os.path.basename(ref).replace("_ref.wav", "").replace(".wav", "")
        if os.path.exists(f"{out_dir}/{name}.safetensors"):
            continue  # already made (resumable)
        prompt, _ = adapt(a, enc, emain, asr, proc, model, meter)
        text, z = prefix_from_reference(a, asr, emain, meter, lm=(model, proc, prompt))
        save_file(
            {"speaker_wavs": prompt.T[None].contiguous(), "prefix_latents": z.contiguous()},
            f"{out_dir}/{name}.safetensors",
            metadata={"sig": "ba7a5a45", "epoch": "300", "prefix_text": text},
        )
        print(f"{name}: prompt {tuple(prompt.shape)}, prefix {len(z) / 12.5:.1f}s '{text[:50]}', {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()

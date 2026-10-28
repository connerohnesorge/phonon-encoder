"""Cloning by audio continuation (as in VALL-E), from raw reference audio only.

The voice prompt comes from the speaker encoder, as before. In addition, the end of the reference
is given to the model as the *start of its own output*: Whisper transcribes the reference, the last
phrases (PREFIX_S seconds) become codec latents (codec-encoder stand-in), and the text is
"<reference transcript> <new sentence>". The frozen flow LM is teacher-forced through the
reference latents and then continues, saying the new sentence in the acoustic context of the real
recording. The reference part is cut from the decoded audio.

Usage: continue_clone.py MODEL_ID OUT_DIR   (eval speakers from eval_clone; writes <spk>_clone<i>.wav)
"""
import os, sys, time

import librosa
import numpy as np
import pyloudnorm
import soundfile as sf
import torch
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration, pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapt_voice import MODELS, REFINE_STEPS, SR, asr_words, refine
from eval_clone import EVAL, TEXTS, speaker_sets, write_ref
from lmprior import LAM as LM_PRIOR, refine_lm
from prep_spk import loud
from refcond import prepare_reference
from train_emain import load_decoder, load_emain
from train_spk import W, load_speaker_encoder

PREFIX_S = float(os.environ.get("PREFIX_S", "4.0"))
TEMP = float(os.environ.get("TEMP", "0.3"))
FLOW_STEPS = int(os.environ.get("FLOW_STEPS", "1"))
EOS_THRESHOLD = 4.0
FRAMES_AFTER_EOS = 2


def prefix_from_reference(a, asr, emain, meter, lm=None):
    """(transcript, normalised-space-ready codec latents) of the reference's last ~PREFIX_S of speech."""
    words = asr_words(asr, a)
    end = words[-1]["timestamp"][1]
    start_i = len(words) - 1
    while start_i > 0 and end - words[start_i - 1]["timestamp"][0] <= PREFIX_S:
        start_i -= 1
    t0 = words[start_i]["timestamp"][0]
    x = a[max(0, int((t0 - 0.05) * SR)) : int((end + 0.15) * SR)]
    x = np.clip(loud(x, -22.0, meter), -1, 1)
    x = np.pad(x, (0, (-len(x)) % 1920))
    with torch.no_grad():
        z = emain(torch.from_numpy(x)[None, None])[0].T
    text = " ".join(w["text"].strip() for w in words[start_i:]).strip()
    if REFINE_STEPS:
        # sharpen the prefix: latents pulled so the decoder reproduces the reference audio
        z = torch.from_numpy(refine(z.numpy(), x))
    if lm is not None and LM_PRIOR:
        # faithful to the reference and likely under the LM (lmprior.py); lm = (model, proc, prompt)
        z = torch.from_numpy(refine_lm(z.numpy(), x, text, lm[2], lm[0], lm[1]))
    return text, z


@torch.no_grad()
def speak(model, proc, prompt, prefix_text, prefix_z, text, gen):
    conv = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": f"{prefix_text} {text}"}]}]
    inp = proc.apply_chat_template(conv, tokenize=True, return_dict=True, add_generation_prompt=True,
                                   processor_kwargs={"audio_prompt_latents": [prompt]})
    n = len(prefix_z)
    tok = model.config.audio_latent_token_id
    ids = torch.cat([inp["input_ids"], torch.full((1, n), tok)], 1)
    zn = (prefix_z - model.latent_mean) / model.latent_std
    out = model(input_ids=ids, audio_prompt_latents=inp["audio_prompt_latents"],
                audio_prompt_latents_mask=inp["audio_prompt_latents_mask"], audio_latents=zn[None] if n else None,
                use_cache=True)
    h, eos, cache = out.last_hidden_state[:, -1], out.logits[:, -1, 0], out.past_key_values
    lats, after = [], None
    max_frames = int(12.5 * (len(text.split()) * 0.6 + 3))
    for _ in range(max_frames):
        noise = torch.randn(1, model.config.latent_dim, generator=gen) * TEMP**0.5
        x = model.sample_latent(h, noise, FLOW_STEPS)
        lats.append(x)
        if after is None and eos.item() > EOS_THRESHOLD:
            after = FRAMES_AFTER_EOS
        if after is not None:
            after -= 1
            if after < 0:
                break
        o = model(input_ids=torch.full((1, 1), tok), audio_latents=x[:, None], past_key_values=cache, use_cache=True)
        h, eos, cache = o.last_hidden_state[:, -1], o.logits[:, -1, 0], o.past_key_values
    gen_z = torch.cat(lats)
    return torch.cat([zn, gen_z]), n


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    repo, rev = MODELS[sys.argv[1]]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    proc = AutoProcessor.from_pretrained(repo, revision=rev)
    model = GradiumPhononTTSForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dec = load_decoder()
    enc = load_speaker_encoder(os.environ.get("ENC", f"{W}/spk_enc_final.safetensors")).eval()
    emain = load_emain(os.environ.get("EMAIN", f"{W}/emain.safetensors")).eval()
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
    meter = pyloudnorm.Meter(SR)
    for s, (ref, _) in speaker_sets().items():
        t = time.time()
        ref_path = f"{out_dir}/{s}_ref.wav"
        write_ref(ref, ref_path)
        a = sf.read(ref_path, dtype="float32")[0]
        with torch.no_grad():
            prompt = enc(torch.from_numpy(prepare_reference(a))[None, None])[0].T
        if os.environ.get("VOICE_DIR"):
            # a prompt adapted to this reference by adapt_voice.py
            from safetensors.torch import load_file
            prompt = load_file(f"{os.environ['VOICE_DIR']}/{s}.safetensors")["speaker_wavs"][0].T.float()
        prefix_text, prefix_z = prefix_from_reference(a, asr, emain, meter, lm=(model, proc, prompt))
        for i, text in enumerate(TEXTS):
            gen = torch.Generator().manual_seed(i)
            z, n = speak(model, proc, prompt, prefix_text, prefix_z, text, gen)
            with torch.no_grad():
                wav = dec.decode(model.denormalize_latents(z)[None]).audio_values[0, 0].numpy()
            sf.write(f"{out_dir}/{s}_clone{i}.wav", wav[n * 1920 :], SR, subtype="FLOAT")
            np.save(f"{out_dir}/{s}_clone{i}.latents.npy", model.denormalize_latents(z).numpy())
        np.save(f"{out_dir}/{s}_prefix.latents.npy", prefix_z.numpy())
        print(f"{s}: prefix '{prefix_text[:60]}' ({len(prefix_z) / 12.5:.1f}s), {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()

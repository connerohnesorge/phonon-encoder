"""Diagnostic: where identity is lost. For held-out real utterances of a speaker, with the adapted
prompt on 40m:
  resynth  decoder(E_main(u))                        -- codec-encoder stand-in alone
  tf       one-step LM predictions, teacher-forced on E_main(u) with u's transcript
  free     LM generation of u's transcript, no audio context
ECAPA cosine vs the speaker's other held-out recordings, divided by the real ceiling.

Usage: diag_tf.py VOICE_DIR SPK [SPK ...]
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
from continue_clone import FLOW_STEPS, TEMP, speak
from eval_clone import speaker_sets
from prep_spk import loud
from train_emain import load_decoder, load_emain
from train_spk import W
from verifiers import Verifiers, load16


@torch.no_grad()
def tf_predict(model, proc, prompt, text, z, gen):
    conv = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": text}]}]
    inp = proc.apply_chat_template(conv, tokenize=True, return_dict=True, add_generation_prompt=True,
                                   processor_kwargs={"audio_prompt_latents": [prompt]})
    n = len(z)
    ids = torch.cat([inp["input_ids"], torch.full((1, n), model.config.audio_latent_token_id)], 1)
    zn = (z - model.latent_mean) / model.latent_std
    out = model(input_ids=ids, audio_prompt_latents=inp["audio_prompt_latents"],
                audio_prompt_latents_mask=inp["audio_prompt_latents_mask"], audio_latents=zn[None])
    h = out.last_hidden_state[0, -n - 1 : -1]
    noise = torch.randn(n, model.config.latent_dim, generator=gen) * TEMP**0.5
    return model.denormalize_latents(model.sample_latent(h, noise, FLOW_STEPS))


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
        res = {"resynth": [], "tf": [], "free": []}
        for i, u in enumerate(held[:3]):
            a = np.clip(loud(sf.read(u["path"], dtype="float32")[0], -22.0, meter), -1, 1)
            a = np.pad(a, (0, (-len(a)) % 1920))
            with torch.no_grad():
                z = emain(torch.from_numpy(a)[None, None])[0].T
                zs = {"resynth": z, "tf": tf_predict(model, proc, prompt, u["text"], z, torch.Generator().manual_seed(i))}
                g, _ = speak(model, proc, prompt, "", torch.zeros(0, z.shape[1]), u["text"], torch.Generator().manual_seed(i))
                zs["free"] = model.denormalize_latents(g)
                for k, zz in zs.items():
                    y = dec.decode(zz[None]).audio_values[0, 0].numpy()
                    e = V.embed(librosa.resample(y, orig_sr=SR, target_sr=16000))["ecapa"]
                    res[k] += [float(e @ real[j]) for j in range(len(real)) if j != i]
        print(s, {k: round(np.mean(v) / ceil, 3) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()

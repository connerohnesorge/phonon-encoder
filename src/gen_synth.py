"""Synthetic (audio, latents) pairs from Phonon itself, for training the codec encoder.

Each item: a catalog voice and a training-set sentence; Phonon generates the codec latents and
the released decoder turns them into audio. Writes shards of `{audio int16, latents f16}`.
"""
import glob, json, os, random, sys, time

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration

W = os.environ.get("PHONON_WORK", "/home/user/work")
REV = "92741c5b07d085ff840e26fea8ce43110b87852f"


def texts():
    out = []
    for f in sorted(glob.glob(f"{W}/data/ltr/*.parquet"))[:3]:
        out += [t for t in pq.read_table(f, columns=["text_normalized"]).column(0).to_pylist() if 20 < len(t) < 160]
    return out


def main(shard, n_items, batch, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    proc = AutoProcessor.from_pretrained("gradium/phonon-31m-en", revision=REV)
    model = GradiumPhononTTSForConditionalGeneration.from_pretrained("gradium/phonon-31m-en", revision=REV, dtype=torch.float32).eval()
    voices = [load_file(p)["speaker_wavs"][0].T.float() for p in sorted(glob.glob(f"{W}/voices/voices/*.safetensors"))]
    T = texts()
    items = []
    t0 = time.time()
    while len(items) < n_items:
        convs, vs = [], []
        for _ in range(batch):
            vs.append(random.choice(voices))
            convs.append([{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": random.choice(T)}]}])
        inputs = proc.apply_chat_template(convs, tokenize=True, return_dict=True, add_generation_prompt=True,
                                          processor_kwargs={"audio_prompt_latents": vs})
        with torch.no_grad():
            out = model.generate(**inputs)
            audio = proc.decode(out.audio_latents, out.audio_latents_mask)
        for i, a in enumerate(audio):
            n = int(out.audio_latents_mask[i].sum())
            lat = out.audio_latents[i, :n]
            a = a.float().numpy()[: n * 1920]
            items.append({"audio": (np.clip(a, -1, 1) * 32767).astype(np.int16), "latents": lat.numpy().astype(np.float16)})
        print(f"{len(items)} items, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(f"{W}/synth", exist_ok=True)
    np.save(f"{W}/synth/shard{shard}.npy", np.array(items, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[1]) + 1234)

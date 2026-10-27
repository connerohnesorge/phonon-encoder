"""Writes a Phonon checkpoint folder with a speaker encoder (`mimi_speaker.*`) added."""
import os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

def write_checkpoint(src_dir, dst_dir, speaker_sd):
    os.makedirs(dst_dir, exist_ok=True)
    for f in ("config.json", "tokenizer.json"):
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    sd = load_file(os.path.join(src_dir, "model.safetensors"))
    sd.update({k: v.contiguous() for k, v in speaker_sd.items()})
    save_file(sd, os.path.join(dst_dir, "model.safetensors"), metadata={"format": "pt"})

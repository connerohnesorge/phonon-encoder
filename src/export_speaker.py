"""Exports a trained speaker encoder into a Phonon checkpoint folder as `mimi_speaker.*`.

The runtime's existing audio path (`ptts --voice ref.wav`, `create_voice --input ref.wav`) then
clones from raw audio with no code change: `MimiEnc` reads these tensors, and the checkpoint's own
`speaker_wavs` projection turns the latents into the voice embedding.

Usage: export_speaker.py SPK_ENC.safetensors OUT_DIR
"""
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from export import write_checkpoint
from mimi_torch import rust_state_dict
from train_spk import load_speaker_encoder

W = os.environ.get("PHONON_WORK", "/home/user/work")

if __name__ == "__main__":
    enc = load_speaker_encoder(sys.argv[1])
    write_checkpoint(f"{W}/model", sys.argv[2], rust_state_dict(enc, "mimi_speaker."))
    print("wrote", sys.argv[2])

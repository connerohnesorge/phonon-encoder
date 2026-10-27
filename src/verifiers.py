"""Three independent speaker verifiers, none used in training:
ECAPA-TDNN (SpeechBrain, VoxCeleb), WavLM-Base-Plus-SV (Microsoft), Resemblyzer (GE2E)."""
import os

import numpy as np
import soundfile as sf
import librosa
import torch

W = os.environ.get("PHONON_WORK", "/home/user/work")


def load16(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    return librosa.resample(a, orig_sr=sr, target_sr=16000) if sr != 16000 else a


class Verifiers:
    names = ("ecapa", "wavlm", "resemblyzer")

    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        from transformers import AutoFeatureExtractor, WavLMForXVector
        from resemblyzer import VoiceEncoder

        self.ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=f"{W}/verif/ecapa")
        self.wavlm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
        self.wavlm_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.rz = VoiceEncoder("cpu", verbose=False)

    @torch.no_grad()
    def embed(self, a16):
        out = {}
        e = self.ecapa.encode_batch(torch.from_numpy(a16)[None]).squeeze()
        out["ecapa"] = (e / e.norm()).numpy()
        inp = self.wavlm_fe(a16, sampling_rate=16000, return_tensors="pt")
        e = self.wavlm(**inp).embeddings[0]
        out["wavlm"] = (e / e.norm()).numpy()
        from resemblyzer import preprocess_wav
        e = self.rz.embed_utterance(preprocess_wav(a16, source_sr=16000))
        out["resemblyzer"] = e / np.linalg.norm(e)
        return out

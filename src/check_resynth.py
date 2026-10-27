"""How much speaker identity survives decoder(codec_encoder(x)) on held-out real speech (ECAPA)."""
import json, os, sys, numpy as np, torch, soundfile as sf, pyloudnorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_emain import load_emain, load_decoder
from prep_spk import loud
from verifiers import Verifiers, load16
import librosa
torch.set_num_threads(1)
W = os.environ.get("PHONON_WORK", "/home/user/work")
enc, dec, V = load_emain(f"{W}/emain.safetensors").eval(), load_decoder(), Verifiers()
meter = pyloudnorm.Meter(24000)
meta = [json.loads(l) for l in open(f"{W}/audio/meta.jsonl")]
for spk in ["vctk_p226", "vctk_p294", "ls_121", "lj"]:
    us = sorted([m for m in meta if m["spk"] == spk and m["dur"] > 3], key=lambda m: m["path"])[20:23]
    sims = []
    for u in us:
        a = loud(sf.read(u["path"], dtype="float32")[0], -22.0, meter)
        with torch.no_grad():
            y = dec.decode(enc(torch.from_numpy(a)[None, None]).transpose(1, 2)).audio_values[0, 0, : len(a)].numpy()
        e1 = V.embed(librosa.resample(a, orig_sr=24000, target_sr=16000))["ecapa"]
        e2 = V.embed(librosa.resample(y, orig_sr=24000, target_sr=16000))["ecapa"]
        sims.append(float(e1 @ e2))
    sf.write(f"{W}/out/resynth_{spk}.wav", y, 24000)
    print(spk, "ECAPA(real, resynth)", np.round(sims, 3))

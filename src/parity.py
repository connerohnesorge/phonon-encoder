import os, sys, subprocess, numpy as np, torch, soundfile as sf, pyloudnorm, librosa
from safetensors.torch import load_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mimi_torch import EncCfg, MimiEncoder, load_moshi, rust_state_dict
from export import write_checkpoint
W = os.environ.get("PHONON_WORK", "/home/user/work")
enc = MimiEncoder(EncCfg()).eval()
miss, unexp = load_moshi(enc, load_file(f"{W}/kmimi/moshi_tokenizer.safetensors"))
print("missing", miss, "unexpected", len(unexp))
write_checkpoint(f"{W}/model", f"{W}/ckpt_parity", rust_state_dict(enc, "mimi_speaker."))
# a test wav: a catalog sample, 24 kHz
a, sr = sf.read(f"{W}/voices/samples/31m/en/characters/azgorath_dGyGDKo2osj7h3s7/line_1.ogg", dtype="float32")
if a.ndim > 1: a = a.mean(1)
a = librosa.resample(a, orig_sr=sr, target_sr=24000) if sr != 24000 else a
sf.write(f"{W}/out/parity.wav", a, 24000, subtype="FLOAT")
subprocess.run([f"{W}/xn-ptts/target/release/examples/create_voice", "--model", f"{W}/ckpt_parity",
                "--input", f"{W}/out/parity.wav", "--output", f"{W}/out/parity_voice.safetensors"], check=True,
               capture_output=True)
rust = load_file(f"{W}/out/parity_voice.safetensors")["emb"][0]
x = a - a.mean()
L = pyloudnorm.Meter(24000).integrated_loudness(x.astype(np.float64))
x = (x * 10 ** ((-18 - L) / 20)).astype(np.float32)[: 240000]
with torch.no_grad():
    lat = enc(torch.from_numpy(x)[None, None])[0].T  # [T, 512]
proj = load_file(f"{W}/model/model.safetensors")["flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight"].float()
emb = lat @ proj.T
print("rust", tuple(rust.shape), "torch", tuple(emb.shape))
n = min(len(rust), len(emb))
d = (rust[:n] - emb[:n]).abs()
print("max abs diff", d.max().item(), "rel", (d.norm() / rust[:n].norm()).item())

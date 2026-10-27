"""Speaker-encoder SEANet features of the synthetic utterances, at the runtime's -18 LUFS."""
import json, os, sys
import numpy as np, pyloudnorm, torch
from safetensors.torch import load_file
sys.path.insert(0, os.path.dirname(__file__))
from mimi_torch import EncCfg, MimiEncoder, load_moshi
from prep_spk import loud
W = os.environ.get("PHONON_WORK", "/home/user/work")
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
enc = MimiEncoder(EncCfg()).eval()
load_moshi(enc, load_file(f"{W}/kmimi/moshi_tokenizer.safetensors"))
meter = pyloudnorm.Meter(24000)
out = []
with torch.no_grad():
    for s in ("shard0", "shard1"):
        for it in np.load(f"{W}/synth/{s}.npy", allow_pickle=True):
            a = np.clip(loud(it["audio"].astype(np.float32) / 32767, -18.0, meter), -1, 1)
            out.append(enc.seanet(torch.from_numpy(a)[None, None])[0].numpy().astype(np.float16))
arr = np.empty(len(out), dtype=object)
for i, x in enumerate(out):
    arr[i] = x
np.save(f"{W}/cache/synfeat.npy", arr, allow_pickle=True)
print("done", len(out))

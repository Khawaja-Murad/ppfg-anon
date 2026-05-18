"""Convert Math-Shepherd PRM .bin shards to .safetensors in-place in the HF cache.

Needed because Math-Shepherd ships only legacy pickle shards, and transformers >=4.50
refuses torch.load() without torch>=2.6 (CVE-2025-32434). Our torch is 2.5.1
(pinned by vLLM 0.7.3 compatibility), so we convert once on the login node.
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file


def main(prm_dir: str) -> None:
    prm_dir = prm_dir.rstrip("/")
    index_path = os.path.join(prm_dir, "pytorch_model.bin.index.json")
    with open(index_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    shards = sorted(set(weight_map.values()))
    print(f"PRM_DIR={prm_dir}")
    print(f"Shards: {shards}")
    new_weight_map: dict[str, str] = {}
    total_size = 0
    for shard in shards:
        bin_path = os.path.realpath(os.path.join(prm_dir, shard))
        print(f"Loading {shard} from {bin_path}", flush=True)
        sd = torch.load(bin_path, map_location="cpu", weights_only=False)
        new_name = shard.replace("pytorch_model", "model").replace(".bin", ".safetensors")
        new_path = os.path.join(prm_dir, new_name)
        sd_contig = {k: v.contiguous() for k, v in sd.items()}
        save_file(sd_contig, new_path, metadata={"format": "pt"})
        size = os.path.getsize(new_path)
        total_size += size
        print(f"Wrote {new_name} ({size / 1e9:.2f} GB)", flush=True)
        for k in sd:
            new_weight_map[k] = new_name
        del sd, sd_contig
    new_idx = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
    new_idx_path = os.path.join(prm_dir, "model.safetensors.index.json")
    with open(new_idx_path, "w") as f:
        json.dump(new_idx, f, indent=2)
    print(f"Wrote {new_idx_path}")
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1])

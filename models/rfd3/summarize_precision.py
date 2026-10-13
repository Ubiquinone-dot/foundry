"""Compare saved precision-audit structures using separate CA/backbone Kabsch fits."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for out in args.runs:
        if not (out / "reference.pt").exists():
            continue
        data = torch.load(out / "input.pt", weights_only=False)
        aa = data["example"]["atom_array"]
        ref = torch.load(out / "reference.pt", weights_only=False)["legacy"]
        n = len(aa)
        refxyz = ref["X_L"][0].numpy()
        selections = {
            "CA": aa.atom_name == "CA",
            "backbone": np.isin(aa.atom_name, ["N", "CA", "C", "O"]),
        }
        results = {}

        def aligned(x, y):
            x = x - x.mean(0)
            y = y - y.mean(0)
            u, _, v = np.linalg.svd(x.T @ y)
            fix = np.eye(3)
            fix[-1, -1] = np.linalg.det(u @ v)
            return float(np.sqrt(np.mean(np.sum((x @ u @ fix @ v - y) ** 2, axis=-1))))

        for path in sorted(out.glob("*.pt")):
            if path.name in ("input.pt", "reference.pt"):
                continue
            state = torch.load(path, weights_only=False)
            state = state.get("legacy", state)
            xyz = state["X_L"].reshape(-1, state["X_L"].shape[-2], 3)[0, :n].numpy()
            item = {
                k + "_aligned_RMSD_A": aligned(xyz[sel], refxyz[sel])
                for k, sel in selections.items()
            }
            ca = xyz[selections["CA"]]
            distances = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
            item.update(
                CA_distance_min_A=float(distances.min()),
                CA_distance_max_A=float(distances.max()),
                finite=bool(np.isfinite(xyz).all()),
                sequence_agreement=float(
                    np.mean(
                        state["sequence_indices_I"]
                        .reshape(-1, state["sequence_indices_I"].shape[-1])[
                            0, : len(ca)
                        ]
                        .numpy()
                        == ref["sequence_indices_I"][0].numpy()
                    )
                ),
            )
            results[path.stem] = item
        ca = refxyz[selections["CA"]]
        d = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
        results["reference_geometry"] = {
            "CA_distance_min_A": float(d.min()),
            "CA_distance_max_A": float(d.max()),
        }
        (out / "structure_comparison.json").write_text(json.dumps(results, indent=2))
        print(out, json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""Audit RFD3 checkpoint numerics with serialized identical inputs and EDM noise.

Run --mode reference with the upstream source on PYTHONPATH, then --mode candidate
with this source. AtomWorks 2.2.1 isolates acceleration from the AtomWorks 3 port.
No checkpoint downloads occur. Only load trusted local checkpoint/snapshot files.

--strict requests deterministic algorithms. Pristine legacy CUDA index_reduce
cannot honor that setting on Torch 2.11. --deterministic-pooling-diagnostic is an
explicit reference intervention replacing that operation with FP32 per-token means;
it does not represent pristine upstream execution or modify repository source.

Output tensors are retained for independent structural/sequence comparisons.
The timing fields include diagnostic transfers/traces and are not speed benchmarks.
"""

import argparse
import copy
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
from rfd3.inference.datasets import ContigJsonDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["reference", "candidate"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--compiled", action="store_true")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--legacy-only", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--label", default="candidate")
    p.add_argument("--deterministic-pooling-diagnostic", action="store_true")
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if a.strict:
        torch.use_deterministic_algorithms(True)
    if a.deterministic_pooling_diagnostic:
        # Explicit audit intervention, not pristine production behavior: index_reduce
        # uses nondeterministic CUDA atomics. Pool each token with an FP32 mean.
        from rfd3.model.layers import blocks

        def deterministic_mean(zeros, dim, index, source):
            pieces = []
            for i in range(zeros.shape[dim]):
                values = source.index_select(dim, (index == i).nonzero().flatten())
                mean = (
                    values.float().mean(dim=dim)
                    if values.shape[dim]
                    else zeros.select(dim, i).float()
                )
                pieces.append(mean.to(zeros.dtype))
            return torch.stack(pieces, dim=dim)

        blocks.scatter_mean = deterministic_mean
    engine = RFD3InferenceEngine(
        **RFD3InferenceConfig(
            ckpt_path=a.checkpoint,
            diffusion_batch_size=1,
            seed=42,
            compile_model=False,
            inference_sampler={"num_timesteps": a.steps, "allow_realignment": False},
            prevalidate_inputs=False,
        )
    )
    engine.initialize()
    net = engine.trainer.state["model"]
    for _ in range(8):
        if hasattr(net, "diffusion_module"):
            break
        for attr in ("_forward_module", "module", "shadow", "model"):
            if hasattr(net, attr):
                net = getattr(net, attr)
                break
    net.eval()
    dm = net.diffusion_module
    sampler = net.inference_sampler.sampler
    # Resolve the existing default to avoid separate compiler graphs for None/2.
    if sampler.n_recycle is None:
        sampler.n_recycle = dm.n_recycle
    for module in net.modules():
        module.inference_kernel_backend = "torch"

    def host(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().clone()
        if isinstance(x, dict):
            return {k: host(v) for k, v in x.items()}
        if isinstance(x, list):
            return [host(v) for v in x]
        return x

    def errors(x, y):
        x, y = x.float(), y.float()
        delta = x - y
        return {
            "max_abs": delta.abs().max().item(),
            "rms": delta.square().mean().sqrt().item(),
            "reference_rms": y.square().mean().sqrt().item(),
            "finite": bool(torch.isfinite(x).all()),
            "exact": bool(torch.equal(x, y)),
        }

    if a.mode == "reference":
        dataset = ContigJsonDataset(
            data={"audit40": {"length": "40"}},
            transform=engine.pipeline,
            name="audit",
            cif_parser_args=None,
            subset_to_keys=None,
            eval_every_n=1,
        )
        ex = dataset[0]
        gen = torch.Generator().manual_seed(919)
        noise = [
            torch.randn(ex["coord_atom_lvl_to_be_noised"].shape, generator=gen)
            for _ in range(a.steps)
        ]
        torch.save({"example": host(ex), "noise": noise}, a.out / "input.pt")
    else:
        data = torch.load(a.out / "input.pt", weights_only=False)
        ex, noise = data["example"], data["noise"]
    ex = engine.trainer.fabric.to_device(ex)
    noise = [v.cuda() for v in noise]
    f = ex["feats"]
    coords = ex["coord_atom_lvl_to_be_noised"]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        init = net.token_initializer(copy.deepcopy(f))
        first = dm(
            coords + noise[0] * 3,
            torch.ones(1, device="cuda") * 2,
            copy.deepcopy(f),
            **init,
            n_recycle=sampler.n_recycle,
        )
        with patch("torch.normal", side_effect=noise):
            legacy = sampler.sample_diffusion_like_af3(
                f=copy.deepcopy(f),
                diffusion_module=dm,
                diffusion_batch_size=1,
                coord_atom_lvl_to_be_noised=coords,
                initializer_outputs=init,
                ref_initializer_outputs=None,
                f_ref=None,
            )
        legacy = host(legacy)
        if a.mode == "reference":
            torch.save(
                {"first": host(first), "legacy": legacy, "init": host(init)},
                a.out / "reference.pt",
            )
            print("REFERENCE_COMPLETE", flush=True)
        else:
            ref = torch.load(a.out / "reference.pt", weights_only=False)
            report = {
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "steps": a.steps,
                "legacy_first": {},
                "legacy_rollout": errors(legacy["X_L"], ref["legacy"]["X_L"]),
            }
            for key in ("X_L", "sequence_logits_I"):
                report["legacy_first"][key] = errors(
                    host(first[key]), ref["first"][key]
                )
            report["initializer"] = {
                k: errors(host(init[k]), v)
                for k, v in ref["init"].items()
                if isinstance(v, torch.Tensor)
            }
            torch.save(
                {"first": host(first), "legacy": legacy, "init": host(init)},
                a.out / (a.label + "_legacy.pt"),
            )
            if a.legacy_only:
                (a.out / (a.label + "_report.json")).write_text(
                    json.dumps(report, indent=2)
                )
                print(json.dumps(report), flush=True)
                raise SystemExit(0)
            from rfd3.inference.batching import collate_examples, prepare_attention
            from rfd3.model.batched_sampler import sample
            from rfd3.model.layers.batched import initialize

            batch = collate_examples(
                [ex], atom_capacity=768, token_capacity=128, slot_capacity=14
            )
            bf = prepare_attention(
                batch["f"],
                atom_keys=dm.n_attn_keys,
                atom_neighbors=dm.n_attn_seq_neighbours,
                token_keys=dm.diffusion_transformer.n_keys,
                token_neighbors=dm.diffusion_transformer.n_local_tokens,
            )
            bi = initialize(net.token_initializer, bf)
            probe_x = torch.zeros_like(batch["coords"])
            probe_x[0, :, : coords.shape[-2]] = coords + noise[0] * 3
            probe_t = torch.full((1, 1), 2.0, device=coords.device)

            def noise_source(tag, template, atomwise=True):
                val = noise[0 if tag == "initial" else int(tag.split("_")[1]) + 1]
                out = torch.zeros_like(template)
                out[0, :, : val.shape[-2]] = val
                return out

            modes = ["padded_eager"] + (["padded_compiled"] if a.compiled else [])
            for mode in modes:
                if mode.endswith("compiled"):
                    engine._compile_diffusion_submodules()
                probe = dm.forward_batched(
                    probe_x, probe_t, bf, **bi, n_recycle=sampler.n_recycle
                )
                probe_errors = {
                    k: errors(
                        host(probe[k][0, :, : ref["first"][k].shape[1]]),
                        ref["first"][k],
                    )
                    for k in ("X_L", "sequence_logits_I")
                }
                del probe
                start = time.monotonic()
                got = sample(
                    sampler, dm, batch["coords"], bf, bi, noise_source=noise_source
                )
                got = host(got)
                n = coords.shape[-2]
                ni = len(f["restype"])
                report[mode] = {
                    "elapsed_s": time.monotonic() - start,
                    "fixed_input_denoiser": probe_errors,
                    "coordinates": errors(got["X_L"][0, :, :n], ref["legacy"]["X_L"]),
                    "logits": errors(
                        got["sequence_logits_I"][0, :, :ni],
                        ref["legacy"]["sequence_logits_I"],
                    ),
                    "trajectory": [],
                }
                for step in (0, 1, 9, 49, 99, a.steps - 2):
                    if step < len(got["X_denoised_L_traj"]):
                        report[mode]["trajectory"].append(
                            {
                                "step": step,
                                **errors(
                                    got["X_denoised_L_traj"][step][0, :, :n],
                                    ref["legacy"]["X_denoised_L_traj"][step],
                                ),
                            }
                        )
                torch.save(got, a.out / (mode + ".pt"))
                (a.out / "report.json").write_text(json.dumps(report, indent=2))
                print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

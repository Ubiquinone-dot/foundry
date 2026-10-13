"""Numerical audit: compare learned denoisers over an identical noise schedule.

A same-seed legacy/batched comparison is invalid: those paths intentionally use
separate RNG contracts. Inject the actual same draws instead, and cover output
logits and every denoised state as well as the final coordinates.
"""

import copy
from unittest.mock import patch

import pytest
import torch
from rfd3.inference.batching import collate_examples, prepare_attention
from rfd3.model.batched_sampler import sample
from rfd3.model.layers.batched import initialize
from rfd3_batch_fixtures import example, tiny_model


@pytest.mark.parametrize("bf16", [False, True])
def test_learned_rollout_same_noise_against_legacy(bf16):
    torch.set_num_threads(2)
    model = tiny_model(bf16=bf16)
    dm = model.diffusion_module
    sampler = model.inference_sampler.sampler
    sampler.num_timesteps = 16
    examples = [example(d=2), example((1, 2, 1, 1), d=2, seed=8)]
    # A fixed motif exercises atom-specific times without geometric alignment.
    examples[1]["feats"]["is_motif_atom_with_fixed_coord"][0] = True
    examples[1]["feats"]["is_motif_token_with_fully_fixed_coord"][0] = True
    batch = collate_examples(
        examples, atom_capacity=10, token_capacity=6, batch_capacity=3
    )
    f = prepare_attention(
        batch["f"],
        atom_keys=dm.n_attn_keys,
        atom_neighbors=dm.n_attn_seq_neighbours,
        token_keys=dm.diffusion_transformer.n_keys,
        token_neighbors=dm.diffusion_transformer.n_local_tokens,
    )
    g = torch.Generator().manual_seed(4321)
    noise = [
        torch.randn(batch["coords"].shape, generator=g)
        for _ in range(sampler.num_timesteps)
    ]

    def source(tag, template, atomwise=True):
        return noise[0 if tag == "initial" else int(tag.split("_")[1]) + 1]

    with torch.no_grad():
        actual = sample(
            sampler,
            dm,
            batch["coords"],
            f,
            initialize(model.token_initializer, f),
            noise_source=source,
        )
        for b, ex in enumerate(examples):
            lf = copy.deepcopy(ex["feats"])
            n, nt = len(lf["atom_to_token_map"]), len(lf["restype"])
            with patch("torch.normal", side_effect=[x[b, :, :n] for x in noise]):
                expected = sampler.sample_diffusion_like_af3(
                    f=lf,
                    diffusion_module=dm,
                    diffusion_batch_size=2,
                    coord_atom_lvl_to_be_noised=ex["coord_atom_lvl_to_be_noised"],
                    initializer_outputs=model.token_initializer(lf),
                    ref_initializer_outputs=None,
                    f_ref=None,
                )
            # FP32 has a tighter bound; BF16 paths reorder rounded reductions.
            tolerance = 0.025 if bf16 else 2e-4
            for key, length in (("X_L", n), ("sequence_logits_I", nt)):
                delta = (actual[key][b, :, :length] - expected[key]).float()
                print(
                    "MATCHED_NOISE",
                    bf16,
                    b,
                    key,
                    "max_abs",
                    delta.abs().max().item(),
                    "rms",
                    delta.square().mean().sqrt().item(),
                )
                assert torch.isfinite(actual[key]).all()
                assert (actual[key][2] == 0).all()
                torch.testing.assert_close(
                    actual[key][b, :, :length],
                    expected[key],
                    atol=tolerance,
                    rtol=tolerance,
                )
            for key in ("X_denoised_L_traj", "sequence_entropy_traj"):
                length = n if key == "X_denoised_L_traj" else nt
                assert len(actual[key]) == len(expected[key]) == 15
                for got, ref in zip(actual[key], expected[key]):
                    torch.testing.assert_close(
                        got[b, :, :length], ref, atol=tolerance, rtol=tolerance
                    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph audit requires CUDA"
)
@pytest.mark.parametrize("bf16", [False, True])
def test_cuda_replay_refreshes_conditioning_masks_and_times(bf16):
    """Replay one shape with changed lengths, motif masks, features and times."""
    model = tiny_model(bf16=bf16).cuda()
    dm = model.diffusion_module
    compiled = torch.compile(
        dm.forward_batched,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": True},
    )
    tolerance = 0.025 if bf16 else 3e-5
    previous = None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        for step in range(4):
            examples = [
                example(
                    (2, 1, 3) if step % 2 == 0 else (1, 2, 1, 1), d=2, seed=100 + step
                )
            ]
            examples[0]["feats"]["is_motif_atom_with_fixed_coord"][0] = bool(step % 2)
            batch = collate_examples(
                examples,
                atom_capacity=10,
                token_capacity=6,
                slot_capacity=3,
                batch_capacity=2,
            )
            f = prepare_attention(
                batch["f"],
                atom_keys=dm.n_attn_keys,
                atom_neighbors=dm.n_attn_seq_neighbours,
                token_keys=dm.diffusion_transformer.n_keys,
                token_neighbors=dm.diffusion_transformer.n_local_tokens,
            )
            f = {k: v.cuda() for k, v in f.items()}
            x = batch["coords"].cuda()
            t = torch.full((2, 2), 0.7 + step * 1.1, device="cuda")
            init = initialize(model.token_initializer, f)
            expected = dm.forward_batched(x, t, f, **init)
            actual = compiled(x, t, f, **init)
            for key in ("X_L", "sequence_logits_I"):
                torch.testing.assert_close(
                    actual[key], expected[key], atol=tolerance, rtol=tolerance
                )
                assert (actual[key][1] == 0).all()
            saved = actual["X_L"].cpu().clone()
            if previous is not None:
                assert not torch.equal(previous, saved)
            previous = saved
            del actual

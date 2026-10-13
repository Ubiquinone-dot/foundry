# Independent PR #397 precision audit (2026-10-13)

This audit found no component-level evidence of cross-example mixing, stale CUDA
replay, padding leakage, or incorrect custom-kernel arithmetic in the exercised
cases. **It does not establish numerical identity or scientific equivalence of
accelerated inference to upstream RFD3.** Approval of a dependency port must remain
separate from approval of the acceleration defaults.

## Provenance and scope

- Pristine upstream: `8bffbcb9bec7eb8c3be0813f87d33c98faa0f5d2`, the parent of PR #397.
- Accelerated source: `754d1caae5a74b35a3bfcf68169e459474b7dbc4`.
- CoreWeave H200, driver 595.71.05; Torch 2.11.0+cu128, Triton 3.6.0;
  AtomWorks 2.2.1, Biotite 1.4.0. No AtomWorks 3 port changes were present.
- Public `rfd3_foundry_2025_12_01_remapped.ckpt`, SHA256
  `9b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1`.
- Native PyTorch RMSNorm fallback, **without Apex**; both arms used that same
  implementation. This is not validation of an Apex-based production image.
- Full-checkpoint test: one unconditional 40-residue example, B=1/D=1,
  200 schedule points, checkpoint recycling, BF16 autocast; padded capacity
  768 atoms/128 tokens/14 slots. No actual motif/ligand checkpoint case or refolding
  was run in this audit. Fixed-atom conditioning was covered by synthetic tests.

## Tests and what they establish

The existing component suite passed **54 tests**, with four CPU-compiler/MPS
opt-in skips. It exercised legacy/batched denoisers, separate example and diffusion
axes, padding/dummy rows, CFG, symmetry, preprocessing RNG/order/sharding, custom
Triton attention/transitions, current weights, and changing-input CUDA graph replay.

Four additional tests in `tests/test_rfd3_matched_noise_audit.py` passed:

1. Learned denoiser rollout with identical injected noise, 16 schedule points,
   two independent examples, D=2, a fixed motif atom, and a dummy row, in FP32 and
   BF16. Coordinates, logits, each denoised state, and entropy trajectories are
   checked. FP32 final coordinate maximum error was 0.0000067 Å; BF16 maximum
   was 0.00399 Å in the CPU mixed-precision fixture.
2. CUDA Inductor replay with changing true lengths, motif masks, conditioning,
   coordinates, and times at fixed capacity, in FP32 and BF16. FP32 comparison
   uses `atol=rtol=3e-5`; BF16 uses `atol=rtol=0.025`.

Those tolerances apply to their explicit fixtures, not arbitrary 200-step
checkpoint rollouts. No sample in this suite establishes biological designability.

## Identical seeds are not an adequate reference

PR #397 enables compilation by default, which selects the padded sampler even
at B=1. `ExampleRandomness` deliberately changes the seed contract: noise is keyed
by global seed, example ID, diffusion-sample index, and event. Preprocessing also
gets a separate per-example seed. Example IDs can contain filenames/prefixes.
Consequently, the same numeric CLI seed across legacy and accelerated paths does
not supply identical inputs/noise.

`benchmark_precision.py` serializes the upstream-prepared example and every
Gaussian draw, then injects exactly those tensors into both samplers. This removes
that confound. It retains output tensors for independent structural comparison.

## Upstream nondeterminism and the diagnostic reference

Even with identical serialized features/noise, pristine upstream repeated in a
fresh process differed by **4.904 Å aligned CA RMSD** (4.831 Å backbone RMSD).
All five initializer tensors were bitwise identical. The accelerated compiled
comparison against that default upstream run was 4.924 Å CA RMSD. Those two numbers
must not be interpreted as proof that the PR causes the observed drift.

Enabling `torch.use_deterministic_algorithms(True)` exposed an upstream operation:
`foundry.utils.torch.scatter_mean` calls CUDA `index_reduce(..., "mean")`, which
has no deterministic implementation in this Torch version. Both pristine
upstream and PR #397's legacy path fail there under strict controls.

A **diagnostic intervention**, `--deterministic-pooling-diagnostic`, replaces only
that legacy pooling operation with explicit per-token FP32 means. It is not
pristine upstream behavior and does not modify production source. Together with
fixed Python hash, cuBLAS workspace, deterministic Inductor, and Torch deterministic
algorithms, upstream and the PR's legacy path then match **bitwise** for all five
initializer tensors, first-step coordinates/logits, and final 200-step coordinates.
This strongly localizes the default legacy replay discrepancy to pooling
nondeterminism in the tested case; it is not a proof of every possible source of
nondeterminism across environments.

The padded path still changes floating-point reduction and matrix-multiplication
order. Against that diagnostic reference, the fixed-input eager padded denoiser
at t=2 differed by maximum 0.0930 Å / coordinate RMS 0.01654 Å. Small local errors
can change nearest-neighbor choices and compound during diffusion. The final eager
padded output differed by **2.372 Å aligned CA RMSD**, with 42.5% sequence-head
agreement. Final-rollout differences must therefore be reported separately from
one-step tolerance checks.

With compilation and CUDA-graph replay enabled under the same controls, the
fixed-input coordinate error was maximum **0.1240 Å**, RMS **0.01934 Å**. The final
structure differed by **2.515 Å aligned CA RMSD** (2.506 Å backbone RMSD), with
**37.5% sequence-head agreement**. This is measured numerical drift, not evidence
that either structure is more designable. Exactness is not established by an
identical seed or by successful execution.

Under the full deterministic controls, a second fresh process reproduced both
padded eager and padded compiled outputs **bitwise**: final coordinates,
sequence logits/indices, and every retained denoised state. This is same-backend
replay in this environment; it does not make either path identical to upstream.

All default-run output coordinates were finite; consecutive CA distances ranged
from 3.765 to 3.871 Å across arms. Those geometry checks reject obvious execution
failure but do not measure foldability, motif preservation, binding, or function.

## Merge interpretation

- **Pass:** the exercised component arithmetic, masks/axes, prefetch ordering,
  cache lifetime, and changing-input replay tests; legacy algorithm agreement
  under the disclosed deterministic reference.
- **Not exact:** the padded/compiled algorithm does not reproduce upstream
  coordinates or sequence-head choices exactly. Default seeds also have a new
  meaning. Fresh-process reproducibility needs explicit controls.
- **Unvalidated:** paired designability against pristine production, actual motif
  geometry under acceleration, nucleic-acid/ligand conditioning, broad lengths and
  seeds, larger independent/diffusion batches with real checkpoints, and Apex.

There is no basis here to call the outputs garbage, but equally no basis to approve
scientific equivalence from component tolerances. Before adopting acceleration by
default, run a paired benchmark against the pristine production revision using a
fixed scorer, task-specific motif/bond constraints, and independent sequence-design
plus refolding or the intended final oracle. Include unconditional, motif/ligand,
and nucleic-acid cases and multiple seeds; set an acceptable quality-loss margin
before looking at results. Keep the known RNG-contract change documented.

PR #397's existing `biological_pilot_2026-10-07.json` is useful supporting evidence
(15/16 backbones passing in both arms), but its documented baseline substitutes
methods from the intermediate fork revision `749d94d` into the newer engine.
It does **not** test the entire PR against pristine production, and it had no
predefined equivalence margin. The PR's own reproducibility record also already
reports a 3.546 Å default fresh-process backbone discrepancy on a motif example.

## Reproducing the controlled comparison

Use separate source checkouts and one dependency environment pinned as above.
The baseline checkout must be selected through `PYTHONPATH`, including both its
`src` and `models/rfd3/src` directories; run the same audit script in both cases.
The script does not download checkpoints.

```bash
PYTHONPATH="$BASE/src:$BASE/models/rfd3/src" "$PYTHON" \
  models/rfd3/benchmark_precision.py --mode reference \
  --checkpoint "$CHECKPOINT" --out "$RUN"
"$PYTHON" models/rfd3/benchmark_precision.py --mode candidate \
  --checkpoint "$CHECKPOINT" --out "$RUN" --compiled
```

For the explicitly modified deterministic diagnostic, set all three environment
variables before either process, and add both flags to each command:

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_DETERMINISTIC=1
# Add: --strict --deterministic-pooling-diagnostic
```

Use `--mode candidate --legacy-only --label upstream_repeat` with the upstream
`PYTHONPATH` to test upstream against itself. Do not interpret reported diagnostic
wall times as speed benchmarks: these runs retain trajectories and transfer data.
Use `python models/rfd3/summarize_precision.py "$RUN"` for aligned CA/backbone
RMSD, adjacent-CA distances, and sequence-head agreement from the saved tensors.
Compact results are in `performance/pr397_precision_audit_2026-10-13.json`.

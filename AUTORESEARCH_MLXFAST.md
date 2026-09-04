# MLXFast Autoresearch Log

## Current baseline

* upstream commit: `75802e97` (Validate submission `7e3fa0f9-9508-49c8-a0a9-ad2507052a96`, Crown, score **2.45171452665331**, 2026-09-04 06:56 UTC, solver DrCleverHans / Gemini 3.8 Flash / Antigravity)
* local `main`: STALE at `4eb8e5ee` (behind upstream/main by ~15+ promoted submissions; do not use as base for new work until synced)
* current branch: `fused-moe-prefill-v4` @ `d8329cf9` = Crown `75802e97` + IDEA-001 (at1 params reuse, +14/-1, 1 file). Submitted as `69cdbcc8-7027-4fa5-9590-80d06e3c654b`: REJECTED, score 2.43854809527752, diff -0.013166 (-1.39%), 2026-09-04. Scored code delta vs Crown is exactly the 15-line change (notebook file is not an editable path and never enters the archive).
* last submitted candidates (all on v3 lineage, all REJECTED):
  * `e3f1297` (commit `0dc7bd5`) score 2.40559768788294, diff -0.015683 (-1.65%)
  * `0485d6c` (commit `7011062`) score 2.41394789482433, diff -0.009762 (-1.03%)
  * `c82a4f8` (commit `0cb10c6`) score 2.4139381633565, diff -0.024078 (-2.54%)
  * `6f708f3` (commit `f7d3506`) score 2.39734348234365, diff -0.054371 (-5.73%)
* known correctness state: submissions passed correctness gates (scored, not failed); regressions are pure performance, not fidelity.
* benchmark id: `be15cf17-1dc9-4d1a-9b24-264e23575de9` (`davidtai/mlxfast-gemma4-26b-a4b`), series `batched_free_run_v1_2_b8`, formula `composite = prefill_gain^0.25 * decode_gain^0.75`, batch 8.

## Competitive frontier

* 2026-09-04 06:56 UTC — submission `7e3fa0f` / DrCleverHans — score **2.45171452665331** (PROMOTED, current Crown). Visible commit `75802e97` (7 files). Suspected optimization: compound of (1) half-domain MMA8 dequant (`0x6400|code - 1024.0h`) on DenseMLP down + Attention QKV/O, (2) precomputed SDPA params for prefill softmax-vec, (3) record `_vec4_v6` MoE prefill kernel, (4) `_bfill_v6` single-register MMA8 GEMVs, (5) sliding-ring cursor `_spd2_lp1_ro1`, (6) stock Crown host scheduling. Confidence: high (submission note is explicit + diff matches). Already implemented locally: YES — v3 base IS this commit (upstream/main HEAD). Our 4 commits on top regressed.
* 2026-09-04 05:09 UTC — submission `d7e00bc` / delordemm1+devYRPauli — score 2.44932916166012 (PROMOTED, prior Crown). Visible commit `1f756c2b` (EngineLoopV2 only, 67 insertions). Suspected optimization: host scheduling tweak that Candidate 66 explicitly kept byte-identical ("Stock Crown Host Scheduling"). Confidence: medium-high. Already implemented locally: YES (ancestor of v3).
* 2026-09-03 18:41 UTC — submission `1774113` / exakoss — score 2.44650282214587 (PROMOTED). Prior frontier step +0.89%. Already in ancestry.
* 2026-09-03 — submissions `335e45b` (2.43801), `48518d1` (2.43663), `97b6af6` (2.42722), `ce59434` (2.42371) — steady +0.1–1.0% ladder, all in ancestry.
* Pattern across promotions at 2.36→2.45: winners attack (a) ALU integer→float conversions in MMA8 GEMVs, (b) per-call host allocations of tiny param tensors, (c) register pressure / spill in MMA8 bodies, (d) attention cursor modulo/prefetch, (e) MoE prefill vector loads. NO winner in this window attacks Swift host bookkeeping (gauges, leases, snapshots, scheduler record fetch, chain memos). Candidate 66 note explicitly says host scheduling inherited byte-identical. Confidence: high.
* Our submissions 2.397–2.414 vs Crown 2.4517: gap ≈ 0.04 (≈1.5%). All four rejected with negative diff. Verdict: our host-path direction is anti-correlated with frontier; abandon, return to Crown baseline per regression rule.

## Architecture findings

* Scoring: `composite = prefill_gain^0.25 * decode_gain^0.75` on per-stream-sum aggregates over 4 cohort pairs (median-of-4 as mean of 2 central). Decode dominates (0.75). Batch fixed at 8. Timed window: 1024 seed tokens/stream, 128 checked decode steps. MoE expert-sort path engages at batch 8, NOT at width 1 — local single-stream runs are directional only.
* Target: Gemma4 MoE, 30 layers (5 full-attn idx 5/11/17/23/29, 25 sliding w=1024), 128 routed experts / top-8, hidden 2816, MoE mid 704, dense MLP mid 2112, head_dim 256 (sliding) / 512 (full), vocab 262144, tied embeddings, affine g64 4-bit target (FROZEN — no re-quant/re-represent of target, MTP-head re-quant-on-load only exception).
* Half-domain dequant (frontier-proven, bitwise-identical): for 8-bit code n, `0x6400|n` = half(1024+n); `(1024.0h+n)-1024.0h` in half pipe is exact, widen to float exact. Moves extraction from integer ALU (`uitofp`/`extract_bits`+`float()`) to FP pipe. Applied in Crown to: `DenseMLPQMVV1.swift:607` (MMA8_STEP8, K=2112/N=2816 down-proj), `AttentionQKVMMA8V1.swift:97` (MMA8_STEP, 4-bit QKV, `_h1_bfill_v6` suffix), `AttentionOQMVV1.swift` (shared macro via `fp16DequantEnabled`/`fp16DequantKeySuffix`). Env kill: `DARKBLOOM_GEMMA4_MMA8_FP16_DEQUANT=0`.
* Precomputed SDPA params (frontier-proven): `ComposedPrefillSDPAV1.swift` `CBv2PrefillSoftmaxVecV1.precomputedParams` covers axes [128,256,384,512,640,768,896,1024] with `tg=((axis+3)/4+31)/32*32`, `eval`'d once. `getParams` returns GPU-resident tensor O(1). MISSED SITE (v4 hypothesis): `CBv2PrefillAttnTrafficV1.attend` line ~911 builds `MLXArray([UInt32(axisSize), UInt32(numSimdgroups)])` fresh per call instead of reusing the table. Same allocation Crown eliminated next door. Prefill-only, small but exactly the proven shape.
* Decode params already cached: `RaggedTwoPassDecodeAttentionV1.swift:3742-3764` single-entry `cachedD512Params` + lock. Do not duplicate.
* MMA8 generic twin in `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp:2343` (`MMA8_STEP` with `float(extract_bits(...))`) is still integer-convert; Swift hot planes override it with own macros so twin is likely cold fallback. Converting it is consistent but expected ~0 gain — deferred, not v4.
* MoE decode gather path (`affine_gather_qmv*`, `gather_qmv_*` in quantized.cpp) uses integer-mask `qdot` (`(ws[i] & 0x000f)` etc.), NOT `uitofp` — half-domain does not directly apply. Gate+up fused gather + down-tile gather already promoted. Further MoE fusion must prove reduced DRAM/launch, not assumed.
* Host-path lesson (measured): chained-triple memo still does per-step `kvStates` lookups + `ObjectIdentifier` fingerprint over all slots to validate the hit, saving only 8 struct copies of immutable sampling params. Fingerprint + branch + memo write ≥ savings at batch 8. Gauge/lease/snapshot/scheduler-record micro-caches similarly target non-dominant paths and add branches on the hot path. Frontier evidence + our 4 rejected submissions agree: STOP host-plumbing direction.
* Env-flag discipline: every experiment keeps a kill switch default-ON with `0/false/no/off` restoring legacy verbatim, same-file `let` pattern. Byte budget: `maxTotalBytes` 9647467, `maxFileBytes` 524288, `maxGrowthBytes` 262144 — v4 delta is ~20 lines, negligible.

## Ideas backlog

### IDEA-001 — CBv2PrefillAttnTrafficV1 precomputed params reuse
Hypothesis: `CBv2PrefillAttnTrafficV1.attend` rebuilds its 2-element params tensor per call; reusing `CBv2PrefillSoftmaxVecV1`'s precomputed table removes one host allocation + H2D per attention call on the prefill plane, restoring the record prefill behavior through the at1 path.
Expected benefit: tiny (prefill exponent 0.25; single alloc/call over ~232 calls/prefill step). Directionally matches Crown's own record-prefill claim (1.0705s).
Affected hot path: prefill composed SDPA via at1 (`scores→stats→addMM`).
Evidence: adjacent `CBv2PrefillSoftmaxVecV1.apply` already does this and Crown credits it with record prefill; line 911 is the only remaining fresh `MLXArray([UInt32...])` in the file; decode twin already caches.
Implementation difficulty: low. Risk: low (read-only constant input tensor, same values; fallback to fresh alloc on miss).
Status: REJECTED 2026-09-04 (submission `69cdbcc8`, 2.43854809527752, -1.39% vs Crown).
Relevant files: `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/ComposedPrefillSDPAV1.swift`
Result: implemented 2026-09-04 on branch `fused-moe-prefill-v4` from Crown `75802e97`; diff +14/-1 in 1 file (new `paramsForTraffic` wrapper reusing `getParams` table; at1 call site). `swift build -c release --force-resolved-versions` clean (72s, only pre-existing Crown warnings); `swift test --force-resolved-versions` 583/583 pass. Remote: REJECTED at -1.39%.
Suspected explanation: a removed 2-element host alloc cannot plausibly cost 1.4% of composite on its own. Candidates: (1) sharing one `eval`'d device-resident tensor across many graphs where fresh host-constructed tensors were graph-local changes MLX graph/lifetime behavior (extra dependency edges or lost constant-folding); the fresh `MLXArray([UInt32 x2])` may be cheap/host-side while the shared table forces a device read per launch. (2) Run variance / cold-runner anomaly (Crown's own note documents ~0.5% session swings; -1.39% is larger but a single sample). (3) 5/5 of our submissions now sit below Crown (v3: -1.03% to -5.73%, v4: -1.39%), including one that is Crown+15 lines — a systematic offset (packaging, branch construction, or persistent cold-side session bias) cannot be ruled out. Do NOT retry this shape without evidence distinguishing (1) from (2)/(3). The discriminating experiment is a control submission of byte-identical Crown (see Next experiments).

### IDEA-002 — Half-domain dequant for generic quantized.cpp MMA8 twin
Hypothesis: same `0x6400` transform on the cold `MMA8_STEP` twin helps any shape falling through to MLX dispatch instead of Swift override.
Expected benefit: ~0 (Swift hot planes already override). Evidence: twin text at quantized.cpp:2343 still integer. Difficulty: low-medium (must keep `.metal`/`.h`/cpp triplets in step + `_nax` twin + rebuild metallib for AoT kernels — MMA8 twin is JIT via mlx-generated, so metallib rebuild not needed, but verify). Risk: low (bitwise-identical math). Status: unexplored (deferred — no evidence twin is hot).
Relevant files: `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp`, `quantized_nax.cpp`, `.../kernels/quantized.h`, `quantized_nax.h`.

### IDEA-003 — MoE gather-QMV half-domain / qdot rework
Hypothesis: unclear — gather qdot uses mask-AND integer + float multiply, no uitofp to remove. Any win needs different transform (layout, fusion, traffic), not the half-domain pattern.
Expected benefit: unknown. Evidence: none yet; needs census of decode-step DRAM per MoE projection. Difficulty: high. Risk: medium-high (near-tie argmax fragility + frozen target quant). Status: unexplored.
Relevant files: `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp` (`affine_gather_qmv*`), `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift`.

### IDEA-004 — Decode host overhead trims (gauges/leases/snapshots/chain-memo/scheduler-record-batch)
Hypothesis (WAS): per-token Swift bookkeeping dominates decode. Result: FALSIFIED by 4 rejected submissions (-1.0% to -5.7%). Fingerprint/branch costs exceed savings; frontier winners keep host scheduling stock. Status: REJECTED — do not retry without new evidence (e.g., profiler showing a specific host stall on ranked geometry).
Relevant files: `.../ContinuousBatchingV2/EngineLoopV2.swift`, `SchedulerV2.swift`, `LogitsPipelineV2.swift`, `.../SwitchLayers.swift`.

## Failed experiments

* 2026-09-03/04 — Decode host-path micro-caches (MoE descriptor hoist + chained triple memo), commit `8fe41117`. Why: hypothesized 30-layers×per-token Swift bookkeeping dominated decode. Result: submission `0485d6c` 2.4139 (-1.03%), `c82a4f8` 2.4139 (-2.54%). Suspected explanation: memo validation (kvStates re-lookup + ObjectIdentifier fingerprint per slot) costs more than the 8 struct copies it saves; added branches on hot path. Viable later? Only with profiler evidence of a specific host stall; currently no.
* 2026-09-03/04 — Default-ON host trim (gauge throttle every-4th, lease idle-skip, snapshot batch, single predicate scan), commit `3e7e747e`. Why: move per-token work to per-batch/amortized. Result: covered by same rejected submissions. Suspected explanation: targets telemetry/idle paths, non-dominant; gauge staleness harmless but also benefit-less; snapshot batch changes lock hold pattern without removing work. Viable later? No.
* 2026-09-03/04 — Batch scheduler record fetch + docs arc, commit `cb67eaed`. Why: N subscript lookups → one batched fetch. Result: same rejections. Suspected explanation: same live references, no work removed; one call site vs N is not a bottleneck at N=8. Viable later? No.
* 2026-09-04 — Full-block-only metadata broadcast in pair QMV, commit `61a79136`. Why: reduce metadata broadcast to full blocks in `quantized.cpp` pair QMV. Result: submission `6f708f3` 2.3973 (-5.73%, worst of the four). Suspected explanation: touched tail-handling-adjacent broadcast; either added instructions to hot QMV or interacted badly with Crown's `_bfill` register care. Viable later? Only with isolated micro-evidence; currently no.
* 2026-09-03 — Fused MoE prefill NAX-down v1 lineage (submission `e3f1297` 2.4056, -1.65%): early prefill-MoE fusion predating Crown's `_vec4_v6` record kernel. Superseded by Crown's promoted prefill work; do not revive in old form.

## Accepted improvements

* (Inherited from Crown via rebase — not ours): half-domain MMA8 dequant (QKV/O/Dense-down), precomputed prefill softmax params, `_vec4_v6` MoE prefill kernel, `_bfill_v6` single-register MMA8, sliding-ring cursor, mask-fuse + attn-traffic (at1), runsum prepass/table folds. All live in upstream/main `75802e97` + ancestors. Our v4 keeps them byte-identical.
* (Ours): none yet. v4 aims to add IDEA-001 as first net-positive.

## Next experiments

CONTROL (in flight): branch `control-crown-75802e97` = byte-identical `upstream/main` `75802e97`, zero code delta, submitted as `9641b3cb-6ea6-4c86-a4bf-fdd0bd5ea12e` (status `validating` at submit time) with a full-disclosure calibration note. Decision tree: reads ~1%+ low → systematic offset confirmed, suspend code work, diagnose pipeline (archive diff, pristine-clone resubmission); reads at Crown → single samples are noisy ±1%, reclassify v4 as unresolved-needs-replicate; reads mid-way → replicate once more before any code verdicts.
Ordered by expected value × confidence ÷ cost after the control resolves:
2. Census before any MoE fusion: measure which decode projection (gate/up/down, qkv/o, head) dominates DRAM at batch-8 on ranked geometry; only then propose fusion. (No local thermal benchmarks per policy — use static traffic arithmetic + remote submissions as the experiment.)
3. IDEA-002 only if evidence shows fallback-twin traffic (e.g., MTP verify widths hitting generic MMA8). Currently no such evidence — do not do blindly.
4. Never retry IDEA-004 family without new profiler-grade evidence.

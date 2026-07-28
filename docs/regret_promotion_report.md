# Regret-Aware Tensor Promotion — off-chip traffic analysis

Re-selects the L2-resident (promoted) tensor set for a training step with the
**regret-aware greedy** of Alg. 1 and reports the **off-chip (L2↔L3) traffic
saved** on five training networks. The existing `PromoteTensorsToL2` mechanism is
unchanged — this PR adds an offline analysis (`scripts/regret_promotion.py`) that
re-does only the *selection* and measures the traffic it eliminates.

## Model

After tiling/checkpointing the live set still exceeds L2, so each step uses both
levels: what fits stays on-chip, the rest spills to L3. Promotion decides that
split. Double-buffering hides all but the first tile of a spilled transfer, so a
spill costs **off-chip traffic** (energy, L3 bandwidth) rather than time.

- `Q(t) = n_acc(t)·|t|` — bytes streamed over L2↔L3 if `t` spills (its reads+writes
  × size); a promoted tensor costs none. This is what promoting `t` **saves**.
- `C(t) = |t|·Σ_{τ∈L(t)} λ_τ` with `λ_τ = (1/B)·Σ_{t' live at τ}|t'|` — the
  **capacity** `t` holds over its lifetime, weighted by contention at each step.
- Feasibility is set-level: whether `t` still fits depends on which rivals were
  admitted before it, so no per-tensor price captures it exactly. That gap is the
  **regret**, approximated by `p_block` (how easily a top-`k` rival would block `t`,
  confirmed with the real allocator) and bounded into the score
  `S(t|P) = Q(t)/C(t)·(1 + γ·p_block(t|P))` so the ranking stays traffic-led.

All three inputs (`|t|`, `n_acc`, live interval `L(t)`) are static graph
properties, extracted directly from the `*_train.onnx`. Feasibility
`MiniMalloc(P) ≤ B` uses the **real** `minimalloc` binary (the same 2D allocator
the tiler uses). Budget `B = C_L2 − 2·C_L1` with `C_L1 ≈ 8%` of L2 reserved for the
double-buffered tile pair; on GAP9 (L2 = 1 MB) that is **B ≈ 840 KB**.

## Results — off-chip traffic saved (per training step, B = 840 KB, γ=1, k=4)

| Network | candidates | traffic if all spill | **regret-aware saved** | `n_acc/ℓ` | `Q`-greedy | `largest` |
|---|--:|--:|--:|--:|--:|--:|
| ResNet8 | 59 | 4.14 MB | **3.89 MB (94.1%)** | 94.1% | 92.3% | 83.6% |
| DSCNN | 49 | 0.15 MB | **0.15 MB (100%)** | 100% | 100% | 100% |
| CCT | 160 | 12.8 MB | **9.59 MB (74.8%)** | 74.3% | 73.0% | 74.2% |
| MCUNet | 270 | 24.0 MB | **11.7 MB (48.9%)** | 51.3% | 36.9% | 31.7% |
| MobileNetV1 | 177 | 9.49 MB | **5.42 MB (57.2%)** | 57.2% | 46.7% | 38.7% |

Regret-aware promotion eliminates **3.89 / 0.15 / 9.59 / 11.7 / 5.42 MB** of
L2↔L3 traffic per step on ResNet8 / DSCNN / CCT / MCUNet / MobileNetV1.

## Findings

1. **Regret-aware is best-or-tied** on ResNet8, CCT, MobileNetV1, and always
   dominates the naive `Q` and `largest` baselines (CCT +1.8pp, MCUNet +12pp,
   MobileNet +10.5pp over `Q`).
2. **One case where it is edged out**: `n_acc/ℓ` beats it on MCUNet (51.3% vs
   48.9%) — worth an ablation row.
3. **The regret boost itself is a bounded nudge**: swept over γ ∈ {1,3,10,30} and
   budgets from 197 KB to 840 KB, it changes only 1–5 greedy picks and leaves the
   saved traffic essentially unchanged (γ=1 and γ=0 are identical on all five
   nets). This is by design — the boost is capped at `1+γ` so the ranking stays
   traffic-led; the savings come from the `Q/C` traffic-per-capacity base ranking
   plus real-`minimalloc` feasibility.

## Reproduce

```bash
export MINIMALLOC_INSTALL_DIR=/path/to/minimalloc
python scripts/regret_promotion.py \
    DeeployTest/Tests/Models/Training/CCT/cct_train/network.onnx \
    --l2 1024000 --gamma 1 -k 4
```

Reports total spillable traffic, the regret-aware selection, and the ablation
baselines (`n_acc`, `n_acc/ℓ`, `Q`, `largest`), plus the `minimalloc` call count.
The analysis is static and needs no gvsoc run; it is deterministic and pure Python
apart from the `minimalloc` binary.

## Scope

`PromoteTensorsToL2` is **not modified** — this is analysis-only. Natural
follow-ups: (a) add `strategy='regret'` to the pass so codegen uses this
selection; (b) a direct delta vs the current `cycle-aware` strategy; (c) sweep γ
and budget to locate the regime where the regret term separates.

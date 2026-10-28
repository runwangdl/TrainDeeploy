# Copyright ETH Zurich 2026
# SPDX-License-Identifier: Apache-2.0
"""exp21b -- aggregate results/<arm>/<subject>.json into tables + plots.

    python3 analyze.py            (any python with numpy + matplotlib; host or container)

Writes results/summary.csv, results/summary_rounds.csv, results/summary.json, results/stability.csv
and plots/*.png.
"""
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, os.environ.get("RESDIR", "results"))
PLOTS = os.path.join(HERE, os.environ.get("PLOTDIR", "plots"))
SUBJECTS = ["S01", "S02", "S03", "S04"]
NSTEP = int(os.environ.get("NSTEP", "1620"))   # update steps per round (120 epochs)
ORDER = ["Z0_no_ft", "A0_qzo_deployed", "A1_sgd_rtn", "A5_sgd_sr", "A10_sgd_carry_rtn", "A11_sgd_carry_sr",
         "A2_sgdm_rtn", "A6_sgdm_sr", "A7_eco_rtn", "A4_eco_sr",
         "A8_adam_rtn", "A3_adam_sr", "A14_adam_rtn_matched", "A12_adam_sr_matched",
         "A13_sgd_sr_lr1e-5"]
OPT = {"sgd": "SGD", "sgdm": "SGDM β0.9", "adam": "Adam β.9/.999"}
EF = {"none": "–", "memoryfree": "ECO mem-free", "carry": "carry (residual)", "exact": "ECO exact"}

# reference palette (dataviz skill, light mode), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def load():
    out = {}
    for arm in ORDER:
        rows = {}
        for s in SUBJECTS:
            f = os.path.join(RES, arm, f"{s}.json")
            if os.path.exists(f):
                rows[s] = json.load(open(f))
        if rows:
            out[arm] = rows
    return out


def factor(r):
    rc = r["recipe"]
    return OPT[rc["optimizer"]], ("SR" if rc["rounding"] == "sr" else "RTN"), EF[rc["ef"]], rc["learning_rate"]


def main():
    os.makedirs(PLOTS, exist_ok=True)
    D = load()
    arms = [a for a in ORDER if a in D]
    summary = {}
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["arm", "optimizer", "rounding", "EF", "lr", "n_subj"] +
                   [f"{s}_mean_b2b5" for s in SUBJECTS] + ["mean_4subj",
                   "moved_step_frac_%", "conv_differ_round_%", "conv_ever_moved_round_%",
                   "conv_differ_base_after_r4_%", "returned_%", "flips_per_step", "net/gross"])
        for a in arms:
            rows = D[a]
            any_r = next(iter(rows.values()))
            o, rd, ef, lr = factor(any_r)
            per = {s: rows[s]["mean_b2_b5"] for s in SUBJECTS if s in rows}
            R = [rr for s in rows for rr in rows[s]["rounds"]]
            agg = dict(
                moved=100 * np.mean([x["moved_step_fraction"] for x in R]),
                differ=100 * np.mean([x["frac_conv_differ_from_round_start"] for x in R]),
                ever=100 * np.mean([x["frac_conv_ever_moved"] for x in R]),
                cum=100 * np.mean([rows[s]["rounds"][-1]["frac_conv_differ_from_base"] for s in rows]),
                returned=100 * np.mean([x["frac_conv_moved_and_returned"] for x in R]),
                fps=np.mean([x["flips_per_step_mean"] for x in R]),
                ng=np.mean([x["net_over_gross"] for x in R]))
            mean4 = float(np.mean(list(per.values()))) if len(per) == 4 else float("nan")
            summary[a] = dict(label=any_r["label"], optimizer=o, rounding=rd, ef=ef, lr=lr,
                              per_subject=per, mean=mean4,
                              acc_by_batch={s: rows[s]["accuracies"] for s in rows}, **agg)
            w.writerow([a, o, rd, ef, lr, len(per)] + [f"{per[s]:.2f}" if s in per else "" for s in SUBJECTS]
                       + [f"{mean4:.2f}", f"{agg['moved']:.1f}", f"{agg['differ']:.2f}", f"{agg['ever']:.2f}",
                          f"{agg['cum']:.2f}", f"{agg['returned']:.2f}", f"{agg['fps']:.1f}", f"{agg['ng']:.3f}"])
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)

    # per-round table (mean over subjects)
    with open(os.path.join(RES, "summary_rounds.csv"), "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["arm", "subject", "round", "eval_acc_before_round", "acc_after(next batch)",
                    "moved_step_%", "differ_round_%", "ever_moved_%", "returned_%", "flipped2+_%",
                    "differ_base_%", "total_flips", "net/gross", "clamped", "bias_differ_%",
                    "train_loss_start", "train_loss_end", "train_acc_start", "train_acc_end"])
        for a in arms:
            for s, r in D[a].items():
                for k, x in enumerate(r["rounds"]):
                    ck = x["checkpoints"]
                    w.writerow([a, s, k + 1, f"{r['accuracies'][k]:.2f}", f"{r['accuracies'][k + 1]:.2f}",
                                f"{100 * x['moved_step_fraction']:.1f}",
                                f"{100 * x['frac_conv_differ_from_round_start']:.2f}",
                                f"{100 * x['frac_conv_ever_moved']:.2f}",
                                f"{100 * x['frac_conv_moved_and_returned']:.2f}",
                                f"{100 * x['frac_conv_flipped_2plus']:.2f}",
                                f"{100 * x['frac_conv_differ_from_base']:.2f}", x["total_flips"],
                                f"{x['net_over_gross']:.3f}", x["clamped_total"],
                                f"{100 * x['frac_bias_differ_from_round_start']:.2f}",
                                f"{ck[0]['train_loss']:.4f}" if ck else "", f"{ck[-1]['train_loss']:.4f}" if ck else "",
                                f"{ck[0]['train_acc']:.2f}" if ck else "", f"{ck[-1]['train_acc']:.2f}" if ck else ""])

    # stability of fixed-lr direct updates: per-step flip statistics, oscillation, curve dips
    with open(os.path.join(RES, "stability.csv"), "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["arm", "steps_with_flip_%", "flips_per_step_mean", "flips_per_step_p99",
                    "max_flips_one_step", "net/gross", "returned_%", "train_acc_drop_max_pp",
                    "next_acc_drop_max_pp", "train_loss_end_minus_start", "float_rel_drift_mean"])
        for a in arms:
            fl, ng, ret, tdrop, ndrop, dl, fd = [], [], [], [], [], [], []
            for s, r in D[a].items():
                for x in r["rounds"]:
                    f = np.asarray(x["per_step"]["flips"]); fl.append(f)
                    ng.append(x["net_over_gross"]); ret.append(x["frac_conv_moved_and_returned"])
                    ck = x["checkpoints"]
                    if ck:
                        ta = np.array([c["train_acc"] for c in ck]); na = np.array([c["next_acc"] for c in ck])
                        tdrop.append(float(np.max(np.maximum.accumulate(ta) - ta)))
                        ndrop.append(float(np.max(np.maximum.accumulate(na) - na)))
                        dl.append(ck[-1]["train_loss"] - ck[0]["train_loss"])
                    fd.append(np.mean(list(x["float_rel_drift"].values())))
            f = np.concatenate(fl)
            if f.size == 0:          # Z0 (no fine-tuning): nothing to report
                continue
            w.writerow([a, f"{100 * (f > 0).mean():.1f}", f"{f.mean():.2f}", f"{np.percentile(f, 99):.0f}",
                        int(f.max()), f"{np.mean(ng):.3f}", f"{100 * np.mean(ret):.2f}",
                        f"{np.max(tdrop) if tdrop else float('nan'):.2f}",
                        f"{np.max(ndrop) if ndrop else float('nan'):.2f}",
                        f"{np.mean(dl) if dl else float('nan'):+.4f}", f"{np.mean(fd):.4f}"])

    # ---- SR-noise repeats (results_srseed/<arm>_sr<k>/): same z/data, only the SR uniforms reseeded
    srdir = os.path.join(HERE, "results_srseed")
    if os.path.isdir(srdir):
        with open(os.path.join(RES, "srseed_repeats.csv"), "w", newline="") as h:
            w = csv.writer(h)
            w.writerow(["arm", "sr_seed"] + SUBJECTS + ["mean_4subj"])
            for a in arms:
                means = []
                for tag, base in [("42 (main)", os.path.join(RES, a))] + \
                        [(str(k), os.path.join(srdir, f"{a}_sr{k}")) for k in (1, 2)]:
                    fs = [os.path.join(base, f"{s}.json") for s in SUBJECTS]
                    if not all(os.path.exists(f) for f in fs):
                        continue
                    v = [json.load(open(f))["mean_b2_b5"] for f in fs]
                    means.append(np.mean(v))
                    w.writerow([a, tag] + [f"{x:.2f}" for x in v] + [f"{np.mean(v):.2f}"])
                if len(means) > 1:
                    w.writerow([a, "mean±std over SR seeds"] + [""] * 4 +
                               [f"{np.mean(means):.2f} ± {np.std(means, ddof=1):.2f} (n={len(means)})"])

    complete = [a for a in arms if len(D[a]) == 4]
    if not complete:
        return

    # ---- plot 1: mean b2-b5 per arm, per-subject dots
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(complete) + 1.2))
    y = np.arange(len(complete))[::-1]
    means = [summary[a]["mean"] for a in complete]
    hl = {"A0_qzo_deployed": C[7], "A1_sgd_rtn": "#f2a7a6", "A4_eco_sr": C[0]}
    ax.barh(y, means, height=0.6, color=[hl.get(a, "#b9b8b2") for a in complete], zorder=2)
    for k, s in enumerate(SUBJECTS):
        ax.scatter([summary[a]["per_subject"][s] for a in complete], y, s=14, color=INK2,
                   marker="osD^"[k], zorder=3, label=s, linewidths=0)
    for yy, m in zip(y, means):
        ax.text(m + 0.3, yy, f"{m:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [f"{a.split('_', 1)[0]}  {summary[a]['label']}" for a in complete], fontsize=8, color=INK)
    lo = min(min(summary[a]["per_subject"].values()) for a in complete)
    ax.set_xlim(lo - 3, 92)
    ax.set_xlabel("balanced accuracy, mean b2–b5 (%)", color=INK2, fontsize=9)
    ax.set_title("exp21b — accuracy per arm (bar = 4-subject mean, markers = subjects)", fontsize=10, color=INK, loc="left")
    ax.axvline(summary.get("A0_qzo_deployed", {}).get("mean", np.nan), color=C[7], lw=1, ls=":")
    style(ax); ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=4)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "accuracy_per_arm.png"), dpi=150); plt.close(fig)

    # ---- plot 2: per-subject accuracy grid (arm x subject heat-table as grouped bars)
    fig, axs = plt.subplots(1, 4, figsize=(13, 0.38 * len(complete) + 1.4), sharey=True)
    for k, s in enumerate(SUBJECTS):
        ax = axs[k]
        v = [summary[a]["per_subject"][s] for a in complete]
        ax.barh(y, v, height=0.6, color=[hl.get(a, "#b9b8b2") for a in complete], zorder=2)
        for yy, m in zip(y, v):
            ax.text(m + 0.2, yy, f"{m:.1f}", va="center", fontsize=7, color=INK)
        ax.set_xlim(min(v) - 4, max(v) + 3); ax.set_title(s, fontsize=10, color=INK)
        style(ax); ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, [a.split("_", 1)[0] + " " + summary[a]["label"] for a in complete], fontsize=7)
    fig.suptitle("mean b2–b5 balanced accuracy per subject", fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "accuracy_per_subject.png"), dpi=150); plt.close(fig)

    # ---- plot 3: convergence curves (mean over subjects), 4 rounds concatenated -- two families
    families = {
        "convergence_curves.png": ["A0_qzo_deployed", "A1_sgd_rtn", "A5_sgd_sr", "A6_sgdm_sr",
                                   "A7_eco_rtn", "A4_eco_sr", "A10_sgd_carry_rtn"],
        "convergence_curves_adam.png": ["A2_sgdm_rtn", "A6_sgdm_sr", "A8_adam_rtn", "A3_adam_sr",
                                        "A12_adam_sr_matched", "A14_adam_rtn_matched", "A13_sgd_sr_lr1e-5"],
    }
    for fname, fam in families.items():
        fam = [a for a in fam if a in complete]
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
        for i, a in enumerate(fam):
            tr_l, tr_a, nx_a, xs = [], [], [], []
            for s in SUBJECTS:
                r = D[a][s]
                l_, ta_, na_, x_ = [], [], [], []
                for k, x in enumerate(r["rounds"]):
                    for c in x["checkpoints"]:
                        l_.append(c["train_loss"]); ta_.append(c["train_acc"]); na_.append(c["next_acc"])
                        x_.append(k * x["total_steps"] + c["step"])
                tr_l.append(l_); tr_a.append(ta_); nx_a.append(na_); xs = x_
            for ax, v in zip(axs, (tr_l, tr_a, nx_a)):
                ax.plot(xs, np.mean(v, 0), color=C[i], lw=1.6, marker="o", ms=2.5,
                        label=f"{a.split('_', 1)[0]} {summary[a]['label']}")
        for ax, t in zip(axs, ("train-subset CE loss (unperturbed)", "train-subset balanced acc (%)",
                               "next-batch balanced acc (%) = what the protocol scores")):
            style(ax); ax.set_title(t, fontsize=9, color=INK, loc="left")
            ax.set_xlabel(f"update step (4 rounds × {NSTEP})", fontsize=8, color=INK2)
            for b in range(1, 4):
                ax.axvline(b * NSTEP, color=INK2, lw=0.6, ls="--")
        h_, l_ = axs[0].get_legend_handles_labels()
        fig.legend(h_, l_, fontsize=8, frameon=False, loc="lower center", ncol=4)
        fig.suptitle(f"convergence, mean over S01–S04 (checkpoint every {NSTEP // 10} steps; dashed = next round = new batch)",
                     fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(PLOTS, fname), dpi=150); plt.close(fig)
    key = families["convergence_curves.png"]
    key = [a for a in key if a in complete]

    # ---- plot 4: % conv weights moved (per round vs round start; cumulative vs base)
    fig, axs = plt.subplots(1, 2, figsize=(13, 0.38 * len(complete) + 1.4), sharey=True)
    per_round = [summary[a]["differ"] for a in complete]
    ever = [summary[a]["ever"] for a in complete]
    cum = [summary[a]["cum"] for a in complete]
    axs[0].barh(y + 0.17, ever, height=0.32, color=C[0], label="moved ≥1× in round", zorder=2)
    axs[0].barh(y - 0.17, per_round, height=0.32, color=C[2], label="differs at round end", zorder=2)
    axs[1].barh(y, cum, height=0.6, color=C[6], zorder=2, label="differs from pretrained after round 4")
    for ax, vals in ((axs[0], ever), (axs[1], cum)):
        for yy, m in zip(y, vals):
            ax.text(m + 0.5, yy, f"{m:.1f}", va="center", fontsize=7, color=INK)
        style(ax); ax.grid(axis="y", visible=False); ax.set_xlim(0, 110)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.set_xlabel("% of 14,880 int8 conv weights", fontsize=8, color=INK2)
    axs[0].set_yticks(y, [a.split("_", 1)[0] + " " + summary[a]["label"] for a in complete], fontsize=7)
    axs[0].set_title("per round (mean over rounds × subjects)", fontsize=9, loc="left")
    axs[1].set_title("cumulative", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "weights_moved.png"), dpi=150); plt.close(fig)

    # ---- plot 5: per-step LSB flips (S01, all rounds), smoothed -- stability of the direct update
    fig, ax = plt.subplots(figsize=(11, 4))
    sel = [a for a in ["A1_sgd_rtn", "A2_sgdm_rtn", "A5_sgd_sr", "A6_sgdm_sr", "A4_eco_sr",
                       "A3_adam_sr", "A12_adam_sr_matched", "A0_qzo_deployed"] if a in complete]
    for i, a in enumerate(sel):
        f = np.concatenate([x["per_step"]["flips"] for x in D[a]["S01"]["rounds"]]).astype(float)
        k = 50
        sm = np.convolve(f, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(len(sm)) + k // 2, np.maximum(sm, 0.05), color=C[i], lw=1.2,
                label=f"{a.split('_', 1)[0]} {summary[a]['label']}")
    ax.set_yscale("log"); style(ax)
    for b in range(1, 4):
        ax.axvline(b * NSTEP, color=INK2, lw=0.6, ls="--")
    ax.set_xlabel("update step", fontsize=8); ax.set_ylabel("conv-weight LSB flips per step (50-step mean; floor 0.05)", fontsize=8)
    ax.set_title("S01 — how many int8 conv weights change per step", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.set_size_inches(11, 5)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "flips_per_step_S01.png"), dpi=150); plt.close(fig)

    # ---- plot 6: per-batch accuracy curves per subject for key arms
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
    for k, s in enumerate(SUBJECTS):
        ax = axs[k]
        for i, a in enumerate(key):
            ax.plot(range(1, 6), D[a][s]["accuracies"], color=C[i], lw=1.4, marker="o", ms=3,
                    label=f"{a.split('_', 1)[0]} {summary[a]['label']}")
        style(ax); ax.set_title(s, fontsize=10); ax.set_xticks(range(1, 6), [f"b{b}" for b in range(1, 6)])
    axs[0].set_ylabel("balanced acc (%)", fontsize=8)
    axs[3].legend(fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "accuracy_per_batch.png"), dpi=150); plt.close(fig)
    print("wrote", RES, PLOTS)


if __name__ == "__main__":
    main()

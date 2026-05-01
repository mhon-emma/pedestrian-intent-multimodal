"""
Ablation B — Speed Stream Permutation
======================================
Question: Is the SF-GRU learning the EGO-VEHICLE'S KINEMATIC TRAJECTORY
(e.g., decelerating as a pedestrian approaches), or is it simply reading
the INSTANTANEOUS SPEED MAGNITUDE at each frame?

These are fundamentally different signals:
  - Kinematic trajectory: "the car slowed from 40 km/h to 5 km/h over 15
    frames" → strong causal cue that the driver saw the pedestrian.
  - Instantaneous magnitude: "the car is currently going slowly" → weaker
    cue; the car could be slow for many unrelated reasons.

Method
------
We randomly shuffle the 14 speed values WITHIN each sample across the time
axis. This preserves the marginal distribution of speed for every sample
(same set of values) but destroys all temporal order — acceleration patterns,
deceleration ramps, and monotonic trends are all scrambled.

All other modalities (local_box, local_context, pose, box) are left unchanged,
so any performance change is ENTIRELY attributable to losing speed's temporal
structure.

We run 15 independent random shuffles and report mean ± std to separate true
signal from lucky/unlucky permutations.

What the result tells us
------------------------
CASE 1 — permuted ≈ baseline (drop < 1%):
    The GRU on the speed stream is learning magnitude, not trajectory.
    A recurrent layer is the wrong architectural choice here.
    Proposed fix:
      (a) Replace the speed GRU layer with explicit hand-crafted kinematic
          features: [v_t, Δv = v_t − v_{t−1}, mean(v), min(v)] as a static
          vector input to the final GRU layer.
      (b) This frees one full GRU layer (256 units each) that could instead
          be used for a richer pose or cross-modal attention module.

CASE 2 — significant drop (>2-3%):
    The GRU IS learning the temporal kinematic pattern. However, it still
    cannot distinguish "decelerating toward pedestrian" from "decelerating
    for a red light" — the model has no scene context about WHY speed changes.
    Proposed fix: fuse ego-speed with a scene-level context (e.g., from
    local_context VGG features) rather than processing speed in isolation
    as the last GRU stage.

Usage
-----
    conda run -n 3ml python ablation_b_speed_perm.py
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ablation_utils import (
    setup_logging, load_test_data, load_model,
    compute_metrics, fmt_metrics,
    RESULTS_DIR,
)

log = setup_logging('ablation_b')

N_TRIALS = 15   # number of independent random permutations
SEED     = 42   # base seed; each trial uses SEED + trial_index for reproducibility

# ── Load ──────────────────────────────────────────────────────────────────────
log.info("Loading test data and model...")
inputs, labels, data_types = load_test_data()
model = load_model()

# Identify the speed input by name so we don't hardcode index 4
speed_idx = data_types.index('speed')
log.info("Speed stream is input index %d (shape %s)", speed_idx, inputs[speed_idx].shape)
log.info("Test set: %d samples | %d crossing (%.1f%%)",
         len(labels), int(labels.sum()), 100 * labels.mean())

# ── Baseline ──────────────────────────────────────────────────────────────────
log.info("\nRunning baseline (speed in original temporal order)...")
baseline_preds   = model.predict(inputs, batch_size=64, verbose=0)
baseline_metrics = compute_metrics(labels, baseline_preds)
log.info("Baseline: %s", fmt_metrics(baseline_metrics))

# ── Permutation trials ────────────────────────────────────────────────────────
log.info("\nRunning %d speed-permutation trials...", N_TRIALS)
trial_results = []

for trial in range(N_TRIALS):
    rng = np.random.default_rng(SEED + trial)

    # Copy all inputs; only modify the speed stream
    permuted = [arr.copy() for arr in inputs]
    speed    = permuted[speed_idx]   # shape (637, 14, 1)

    # Shuffle time dimension independently for each sample so that
    # per-sample speed values are preserved but their order is randomised
    for sample_i in range(speed.shape[0]):
        perm_idx = rng.permutation(speed.shape[1])
        speed[sample_i] = speed[sample_i][perm_idx]

    permuted[speed_idx] = speed
    preds   = model.predict(permuted, batch_size=64, verbose=0)
    metrics = compute_metrics(labels, preds)
    trial_results.append(metrics)
    log.info("  Trial %2d: %s", trial + 1, fmt_metrics(metrics))

# ── Aggregate across trials ───────────────────────────────────────────────────
accs = np.array([r['acc'] for r in trial_results])
f1s  = np.array([r['f1']  for r in trial_results])
aucs = np.array([r['auc'] for r in trial_results])

summary = dict(
    acc_mean=accs.mean(), acc_std=accs.std(),
    f1_mean=f1s.mean(),   f1_std=f1s.std(),
    auc_mean=aucs.mean(), auc_std=aucs.std(),
)

acc_drop = baseline_metrics['acc'] - summary['acc_mean']
f1_drop  = baseline_metrics['f1']  - summary['f1_mean']

log.info("\n--- Summary ---")
log.info("Baseline  acc=%.4f  f1=%.4f  auc=%.4f",
         baseline_metrics['acc'], baseline_metrics['f1'], baseline_metrics['auc'])
log.info("Permuted  acc=%.4f±%.4f  f1=%.4f±%.4f  auc=%.4f±%.4f",
         summary['acc_mean'], summary['acc_std'],
         summary['f1_mean'],  summary['f1_std'],
         summary['auc_mean'], summary['auc_std'])
log.info("Drop      acc Δ=%.4f (%.2f%%)  |  f1 Δ=%.4f (%.2f%%)",
         acc_drop, 100 * acc_drop, f1_drop, 100 * f1_drop)

log.info("\n--- Interpretation ---")
if abs(acc_drop) < 0.01:
    log.info("FINDING: Speed temporal ordering is IRRELEVANT to the model.")
    log.info("  The GRU on the speed stream reads instantaneous magnitude, not")
    log.info("  the kinematic trajectory (deceleration / acceleration pattern).")
    log.info("  → Proposed fix: replace speed GRU with explicit features")
    log.info("    [v_t, Δv, mean(v)] as a static input — saves one GRU layer.")
else:
    log.info("FINDING: Permuting speed DOES hurt performance (Δacc=%.2f%%).",
             100 * acc_drop)
    log.info("  The GRU captures some temporal speed pattern, but it cannot")
    log.info("  contextualise WHY speed changes (red light vs. yielding to ped).")
    log.info("  → Proposed fix: cross-attend speed stream with local_context")
    log.info("    features so the model knows WHAT the car is decelerating for.")

# ── Save ──────────────────────────────────────────────────────────────────────
save_path = os.path.join(RESULTS_DIR, 'ablation_b_results.pkl')
with open(save_path, 'wb') as f:
    pickle.dump({'trials': trial_results, 'summary': summary,
                 'baseline': baseline_metrics}, f)
log.info("\nResults saved to %s", save_path)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
metric_keys   = ['acc',      'f1',      'auc']
metric_labels = ['Accuracy', 'F1 Score', 'AUC']
perm_arrays   = [accs, f1s, aucs]
perm_means    = [summary['acc_mean'], summary['f1_mean'], summary['auc_mean']]
perm_stds     = [summary['acc_std'],  summary['f1_std'],  summary['auc_std']]
base_vals     = [baseline_metrics['acc'], baseline_metrics['f1'], baseline_metrics['auc']]

for ax, mlabel, pv, pm, ps, bv in zip(
        axes, metric_labels, perm_arrays, perm_means, perm_stds, base_vals):

    x = np.arange(N_TRIALS)

    # Scatter individual trial results
    ax.scatter(x, pv, color='tomato', s=55, zorder=4, label='Permuted trials')

    # Permuted mean ± 1 std band
    ax.axhline(pm, color='tomato', ls='-', lw=2,
               label=f'Permuted mean ({pm:.3f})')
    ax.fill_between([-1, N_TRIALS], pm - ps, pm + ps,
                    alpha=0.15, color='tomato', label='±1 std')

    # Baseline
    ax.axhline(bv, color='steelblue', ls='--', lw=2.2,
               label=f'Baseline ({bv:.3f})')

    ax.set_xlabel('Trial index', fontsize=11)
    ax.set_ylabel(mlabel, fontsize=11)
    ax.set_title(f'{mlabel}: Baseline vs. Permuted Speed', fontsize=11)
    ax.set_xlim(-0.5, N_TRIALS - 0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    'Ablation B: Speed Stream Permutation\n'
    'If permuted ≈ baseline, the speed GRU learns magnitude, not kinematic trajectory.',
    fontsize=12, y=1.02,
)
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'ablation_b.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close(fig)
log.info("Plot saved to %s", plot_path)

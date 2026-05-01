"""Ablation B — Box Stream Permutation (variant of ablation_b_speed_perm.py)"""

import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ablation_utils import (
    setup_logging, load_test_data, load_model,
    compute_metrics, fmt_metrics, RESULTS_DIR,
)

log = setup_logging('ablation_b_box')

N_TRIALS = 15
SEED     = 42

log.info("Loading test data and model...")
inputs, labels, data_types = load_test_data()
model = load_model()

box_idx = data_types.index('box')
log.info("Box stream is input index %d (shape %s)", box_idx, inputs[box_idx].shape)

log.info("\nRunning baseline...")
baseline_preds   = model.predict(inputs, batch_size=64, verbose=0)
baseline_metrics = compute_metrics(labels, baseline_preds)
log.info("Baseline: %s", fmt_metrics(baseline_metrics))

log.info("\nRunning %d box-permutation trials...", N_TRIALS)
trial_results = []

for trial in range(N_TRIALS):
    rng = np.random.default_rng(SEED + trial)
    permuted = [arr.copy() for arr in inputs]
    box = permuted[box_idx]
    for sample_i in range(box.shape[0]):
        perm_idx = rng.permutation(box.shape[1])
        box[sample_i] = box[sample_i][perm_idx]
    permuted[box_idx] = box
    preds   = model.predict(permuted, batch_size=64, verbose=0)
    metrics = compute_metrics(labels, preds)
    trial_results.append(metrics)
    log.info("  Trial %2d: %s", trial + 1, fmt_metrics(metrics))

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

save_path = os.path.join(RESULTS_DIR, 'ablation_b_box_perm.pkl')
with open(save_path, 'wb') as f:
    pickle.dump({'trials': trial_results, 'summary': summary,
                 'baseline': baseline_metrics}, f)
log.info("Results saved to %s", save_path)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
metric_labels = ['Accuracy', 'F1 Score', 'AUC']
perm_arrays   = [accs, f1s, aucs]
perm_means    = [summary['acc_mean'], summary['f1_mean'], summary['auc_mean']]
perm_stds     = [summary['acc_std'],  summary['f1_std'],  summary['auc_std']]
base_vals     = [baseline_metrics['acc'], baseline_metrics['f1'], baseline_metrics['auc']]

for ax, mlabel, pv, pm, ps, bv in zip(axes, metric_labels, perm_arrays, perm_means, perm_stds, base_vals):
    x = np.arange(N_TRIALS)
    ax.scatter(x, pv, color='tomato', s=55, zorder=4, label='Permuted trials')
    ax.axhline(pm, color='tomato', ls='-', lw=2, label=f'Permuted mean ({pm:.3f})')
    ax.fill_between([-1, N_TRIALS], pm - ps, pm + ps, alpha=0.15, color='tomato', label='±1 std')
    ax.axhline(bv, color='steelblue', ls='--', lw=2.2, label=f'Baseline ({bv:.3f})')
    ax.set_xlabel('Trial index', fontsize=11)
    ax.set_ylabel(mlabel, fontsize=11)
    ax.set_title(f'{mlabel}: Baseline vs. Permuted Box', fontsize=11)
    ax.set_xlim(-0.5, N_TRIALS - 0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Ablation B: Box Stream Permutation', fontsize=12, y=1.02)
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'ablation_b_box.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close(fig)
log.info("Plot saved to %s", plot_path)

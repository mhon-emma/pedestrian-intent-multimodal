"""
Ablation D — Saliency with box+speed only (local_box, local_context, pose zeroed).
Input order: [local_box(0), local_context(1), pose(2), box(3), speed(4)]
"""

import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ablation_utils import (
    setup_logging, load_test_data, load_model,
    compute_metrics, fmt_metrics, RESULTS_DIR, MODALITY_COLORS,
)

log = setup_logging('ablation_d_box_speed')
N_FRAMES = 14

log.info("Loading test data and model...")
inputs, labels, data_types = load_test_data()
model = load_model()

# Zero out local_box(0), local_context(1), pose(2)
zeroed = [arr.copy() for arr in inputs]
zeroed[0] = np.zeros_like(inputs[0])
zeroed[1] = np.zeros_like(inputs[1])
zeroed[2] = np.zeros_like(inputs[2])
log.info("Zeroed streams: local_box, local_context, pose")

log.info("\nBaseline (box+speed only)...")
baseline_preds   = model.predict(zeroed, batch_size=64, verbose=0)
baseline_metrics = compute_metrics(labels, baseline_preds)
log.info("Baseline: %s", fmt_metrics(baseline_metrics))

log.info("\nComputing gradients...")
import keras.backend as K
mean_output  = K.mean(model.output)
grad_tensors = K.gradients(mean_output, model.inputs)
grad_fn      = K.function(model.inputs + [K.learning_phase()], grad_tensors)
raw_grads    = grad_fn(list(zeroed) + [0])

saliency = {}
for i, dtype in enumerate(data_types):
    abs_g = np.abs(raw_grads[i])
    saliency[dtype] = abs_g.mean(axis=(0, 2))

total_sal = sum(v.sum() for v in saliency.values())
fracs = {dt: saliency[dt].sum() / total_sal for dt in data_types}
late_fracs = {dt: saliency[dt][-3:].sum() / (saliency[dt].sum() + 1e-12)
              for dt in data_types}

log.info("\n--- Modality contribution ---")
for dt in data_types:
    log.info("  %-15s  %.1f%%  (last-3-frames: %.1f%%)",
             dt, 100*fracs[dt], 100*late_fracs[dt])

save_path = os.path.join(RESULTS_DIR, 'ablation_d_box_speed.pkl')
with open(save_path, 'wb') as f:
    pickle.dump({'saliency': saliency, 'fracs': fracs,
                 'late_fracs': late_fracs, 'baseline': baseline_metrics}, f)
log.info("Results saved to %s", save_path)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 8))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)
frame_labels = [f't-{N_FRAMES-1-i}' for i in range(N_FRAMES)]

ax_heat = fig.add_subplot(gs[0, :])
sal_matrix = np.array([saliency[dt] for dt in data_types])
sal_norm   = sal_matrix / (sal_matrix.max(axis=1, keepdims=True) + 1e-12)
im = ax_heat.imshow(sal_norm, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax_heat.set_yticks(range(len(data_types)))
ax_heat.set_yticklabels(data_types, fontsize=10)
ax_heat.set_xticks(range(N_FRAMES))
ax_heat.set_xticklabels(frame_labels, fontsize=8, rotation=45)
ax_heat.set_title('Gradient Saliency Heatmap — Box+Speed Only', fontsize=11)
plt.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.02)
for row in range(sal_norm.shape[0]):
    for col in range(sal_norm.shape[1]):
        tc = 'white' if sal_norm[row, col] > 0.65 else 'black'
        ax_heat.text(col, row, f'{sal_norm[row, col]:.2f}',
                     ha='center', va='center', fontsize=6.5, color=tc)

ax_bar = fig.add_subplot(gs[1, 0])
bar_colors = [MODALITY_COLORS[dt] for dt in data_types]
bars = ax_bar.bar(data_types, [fracs[dt] for dt in data_types],
                  color=bar_colors, edgecolor='white', linewidth=0.8)
ax_bar.bar_label(bars, labels=[f'{fracs[dt]:.1%}' for dt in data_types],
                 padding=3, fontsize=9)
ax_bar.set_ylabel('Fraction of total saliency', fontsize=10)
ax_bar.set_title('Modality Contribution (box+speed only)', fontsize=11)
ax_bar.tick_params(axis='x', rotation=20)
ax_bar.grid(axis='y', alpha=0.3)
ax_bar.set_ylim(0, max(fracs.values()) * 1.25)

ax_line = fig.add_subplot(gs[1, 1])
for dtype in data_types:
    sal_n = saliency[dtype] / (saliency[dtype].max() + 1e-12)
    ax_line.plot(range(N_FRAMES), sal_n, 'o-',
                 color=MODALITY_COLORS[dtype], lw=2, ms=5, label=dtype)
ax_line.axvspan(11, 13.4, alpha=0.08, color='red', label='Last 3 frames')
ax_line.set_xticks(range(N_FRAMES))
ax_line.set_xticklabels(frame_labels, fontsize=7, rotation=45)
ax_line.set_title('Temporal Saliency Profile (box+speed only)', fontsize=11)
ax_line.legend(fontsize=8, ncol=2, loc='upper left')
ax_line.grid(True, alpha=0.3)

plt.suptitle('Ablation D: Saliency — Box+Speed Only', fontsize=13)
plot_path = os.path.join(RESULTS_DIR, 'ablation_d_box_speed.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close(fig)
log.info("Plot saved to %s", plot_path)

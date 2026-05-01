"""
Ablation D — Per-Frame Gradient Saliency
==========================================
Question: Which frames and which modalities is the SF-GRU actually
attending to when it makes a prediction?

The model gives no explicit attention weights (unlike PIEPredict, which
has a temporal attention mechanism). We recover implicit attention via
input-gradient saliency: how much does a small change to frame t of
modality m affect the output?

Method
------
We compute the gradient of the mean model output with respect to every
input tensor:

    S[m, t] = E_samples[ mean_feat | d(output) / d(input[m, t, :]) | ]

Concretely:
  1. Build a Keras gradient function:  grad_fn = K.function(inputs, d(mean_output)/d(inputs))
  2. Pass all test samples in one forward+backward pass.
  3. Take the absolute value (we care about magnitude, not sign).
  4. Average over the sample axis (637 samples) and the feature axis
     (512 for VGG, 36 for pose, 4 for box, 1 for speed) to get a
     scalar per (modality, frame) cell.

This gives a (5 modalities × 14 frames) saliency matrix.

Note on K.gradients vs. per-sample gradients
---------------------------------------------
K.gradients(K.mean(output), inputs) computes the gradient of the BATCH MEAN
output. This is equivalent to the sample-average gradient and is numerically
stable and efficient. It does NOT give per-sample saliency maps — for that
you'd need to loop over samples, which is 637× slower. For architectural
claims about aggregate model behaviour, the batch-mean gradient is sufficient
and standard in the literature.

What the result tells us
------------------------
FINDING 1 — Temporal concentration (most saliency in last 1-3 frames):
    The GRU's forward accumulation is overwriting early context. Hidden
    state at frame 14 contains little information about frame 1.
    Proposed fix: residual / skip connections from each frame's input
    directly to the classifier, or a Transformer that attends globally.

FINDING 2 — Modality imbalance (one modality dominates saliency):
    The dominant modality is the bottleneck; the others are redundant
    in the current fusion scheme. Two possibilities:
      (a) Weak modalities are being suppressed by the GRU's sequential
          concatenation — the hidden state at later GRU layers has already
          been shaped by earlier modalities and the later ones can't override.
      (b) Some modalities genuinely carry less information about intention.
    Either way, a cross-attention fusion (each modality queries the others)
    would let the model learn the right balance dynamically.

FINDING 3 — Low saliency for pose throughout all frames:
    Pose (36-dim keypoints) enters at the 3rd GRU layer of 5. By then the
    GRU state is already dominated by local_box and local_context features.
    Pose is being starved of influence by the fixed fusion order.
    Proposed fix: inject pose at the first layer (highest priority) or
    use a parallel fusion with learned weighting.

Usage
-----
    conda run -n 3ml python ablation_d_saliency.py
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ablation_utils import (
    setup_logging, load_test_data, load_model,
    compute_metrics, fmt_metrics,
    RESULTS_DIR, MODALITY_COLORS,
)

log = setup_logging('ablation_d')

N_FRAMES = 14   # 15 - 1 due to normalize_boxes=True

# ── Load ──────────────────────────────────────────────────────────────────────
log.info("Loading test data and model...")
inputs, labels, data_types = load_test_data()
model = load_model()
log.info("Test set: %d samples | modalities: %s", len(labels), data_types)

# Confirm baseline so we know the model is working before touching gradients
log.info("\nConfirming baseline inference...")
baseline_preds   = model.predict(inputs, batch_size=64, verbose=0)
baseline_metrics = compute_metrics(labels, baseline_preds)
log.info("Baseline: %s", fmt_metrics(baseline_metrics))

# ── Gradient computation ──────────────────────────────────────────────────────
log.info("\nBuilding Keras gradient function...")

import keras.backend as K

# d( mean over batch of sigmoid output ) / d( each input tensor )
# Result: a list of gradient arrays, one per input, each shape (N, 14, feat_dim)
mean_output  = K.mean(model.output)
grad_tensors = K.gradients(mean_output, model.inputs)

# K.function wraps a TF session call.
# model.inputs is the list of symbolic input tensors.
# K.learning_phase() = 0 means inference mode (no dropout active).
grad_fn = K.function(model.inputs + [K.learning_phase()], grad_tensors)

log.info("Running gradient computation (one forward + backward pass)...")
raw_grads = grad_fn(list(inputs) + [0])
# raw_grads: list of 5 arrays, each shape (637, 14, feat_dim)

# ── Aggregate: mean |gradient| over samples and feature dims ──────────────────
# Shape after aggregation: (14,) per modality — one scalar per (modality, frame)
saliency = {}
for i, dtype in enumerate(data_types):
    abs_g = np.abs(raw_grads[i])              # (637, 14, feat_dim)
    saliency[dtype] = abs_g.mean(axis=(0, 2)) # (14,)  — avg over samples and features

log.info("\n--- Per-modality mean saliency (14 frames) ---")
for dtype in data_types:
    log.info("  %-15s  %s",
             dtype,
             np.array2string(saliency[dtype], precision=5, suppress_small=True))

# ── Analysis ──────────────────────────────────────────────────────────────────
total_sal = sum(v.sum() for v in saliency.values())

log.info("\n--- Relative modality contribution (% of total saliency mass) ---")
fracs = {}
for dtype in data_types:
    fracs[dtype] = saliency[dtype].sum() / total_sal
    log.info("  %-15s  %.1f%%", dtype, 100 * fracs[dtype])

log.info("\n--- Temporal concentration: fraction of saliency in LAST 3 frames ---")
late_fracs = {}
for dtype in data_types:
    late_fracs[dtype] = saliency[dtype][-3:].sum() / saliency[dtype].sum()
    log.info("  %-15s  %.1f%% of saliency in frames 11-13 (latest)", dtype,
             100 * late_fracs[dtype])

dominant = max(fracs, key=fracs.get)
weakest  = min(fracs, key=fracs.get)
pose_frac = fracs.get('pose', 0.0)
mean_late = np.mean(list(late_fracs.values()))

log.info("\n--- Interpretation ---")
log.info("Dominant modality : %s (%.1f%% of saliency)", dominant, 100*fracs[dominant])
log.info("Weakest modality  : %s (%.1f%% of saliency)", weakest,  100*fracs[weakest])
log.info("Mean saliency in last 3 frames: %.1f%%", 100*mean_late)

if mean_late > 0.60:
    log.info("FINDING 1: Saliency is heavily concentrated in the last few frames.")
    log.info("  The GRU overwrites early context; the stacked recurrent design")
    log.info("  does not benefit from long observation windows.")
    log.info("  → Proposed fix: residual connections or Transformer encoder.")

if fracs[dominant] > 0.50:
    log.info("FINDING 2: '%s' dominates >50%% of saliency.", dominant)
    log.info("  Other modalities are largely ignored — the sequential concat")
    log.info("  fusion allows early modalities to monopolise GRU state.")
    log.info("  → Proposed fix: cross-attention between modalities so each")
    log.info("    stream can query all others equally.")

if pose_frac < 0.10:
    log.info("FINDING 3: Pose contributes only %.1f%% of saliency.", 100*pose_frac)
    log.info("  Pose enters the network at GRU layer 3 of 5. By then the state")
    log.info("  is already dominated by VGG features (local_box, local_context)")
    log.info("  and pose cannot override the accumulated representation.")
    log.info("  → Proposed fix: move pose to the first GRU layer, or use a")
    log.info("    parallel-branch design with a learned fusion gate.")

# ── Save ──────────────────────────────────────────────────────────────────────
save_path = os.path.join(RESULTS_DIR, 'ablation_d_results.pkl')
with open(save_path, 'wb') as f:
    pickle.dump({'saliency': saliency, 'fracs': fracs,
                 'late_fracs': late_fracs, 'baseline': baseline_metrics}, f)
log.info("\nResults saved to %s", save_path)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 8))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)
frame_labels = [f't-{N_FRAMES-1-i}' for i in range(N_FRAMES)]

# --- D1: Saliency heatmap (modalities × frames) ---
ax_heat = fig.add_subplot(gs[0, :])

# Normalise per modality row so each row fills [0,1] independently —
# this makes the temporal pattern visible even for low-saliency modalities
sal_matrix = np.array([saliency[dt] for dt in data_types])  # (5, 14)
sal_norm   = sal_matrix / (sal_matrix.max(axis=1, keepdims=True) + 1e-12)

im = ax_heat.imshow(sal_norm, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax_heat.set_yticks(range(len(data_types)))
ax_heat.set_yticklabels(data_types, fontsize=10)
ax_heat.set_xticks(range(N_FRAMES))
ax_heat.set_xticklabels(frame_labels, fontsize=8, rotation=45)
ax_heat.set_xlabel('Observation frame  (t-13 = earliest → t-0 = latest)', fontsize=10)
ax_heat.set_title('Gradient Saliency Heatmap (row-normalised — shows temporal pattern per modality)',
                  fontsize=11)
plt.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.02,
             label='Normalised saliency (per-modality)')

# Annotate cells with values
for row in range(sal_norm.shape[0]):
    for col in range(sal_norm.shape[1]):
        text_color = 'white' if sal_norm[row, col] > 0.65 else 'black'
        ax_heat.text(col, row, f'{sal_norm[row, col]:.2f}',
                     ha='center', va='center', fontsize=6.5, color=text_color)

# --- D2: Modality contribution pie/bar ---
ax_bar = fig.add_subplot(gs[1, 0])
bar_colors = [MODALITY_COLORS[dt] for dt in data_types]
bars = ax_bar.bar(data_types, [fracs[dt] for dt in data_types],
                  color=bar_colors, edgecolor='white', linewidth=0.8)
ax_bar.bar_label(bars, labels=[f'{fracs[dt]:.1%}' for dt in data_types],
                 padding=3, fontsize=9)
ax_bar.set_ylabel('Fraction of total saliency', fontsize=10)
ax_bar.set_title('Modality Contribution\n(% of global saliency mass)', fontsize=11)
ax_bar.tick_params(axis='x', rotation=20)
ax_bar.grid(axis='y', alpha=0.3)
ax_bar.set_ylim(0, max(fracs.values()) * 1.25)

# --- D3: Temporal saliency profile (all modalities, overlaid line chart) ---
ax_line = fig.add_subplot(gs[1, 1])
for dtype in data_types:
    sal_n = saliency[dtype] / (saliency[dtype].max() + 1e-12)
    ax_line.plot(range(N_FRAMES), sal_n, 'o-',
                 color=MODALITY_COLORS[dtype], lw=2, ms=5, label=dtype)

# Highlight the "last 3 frames" zone
ax_line.axvspan(11, 13.4, alpha=0.08, color='red', label='Last 3 frames')
ax_line.set_xticks(range(N_FRAMES))
ax_line.set_xticklabels(frame_labels, fontsize=7, rotation=45)
ax_line.set_xlabel('Observation frame', fontsize=10)
ax_line.set_ylabel('Normalised saliency', fontsize=10)
ax_line.set_title('Temporal Saliency Profile per Modality', fontsize=11)
ax_line.legend(fontsize=8, ncol=2, loc='upper left')
ax_line.grid(True, alpha=0.3)

plt.suptitle(
    'Ablation D: Per-Frame Gradient Saliency\n'
    'Reveals which frames and modalities the model actually uses',
    fontsize=13,
)
plot_path = os.path.join(RESULTS_DIR, 'ablation_d.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close(fig)
log.info("Plot saved to %s", plot_path)

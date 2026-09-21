"""
mixup_augmentation.py
========================
Investigation: synthetic minority-class / cross-domain augmentation for
JAAD's small training set (195 behavior-annotated samples), the clearest
evidenced bottleneck in this project -- JAAD-side training has shown
high seed variance everywhere it's been tried (e.g. GRL+contrastive
JAAD->PIE: 0.583+/-0.073 n=3; full-combo JAAD->PIE seeds ranged
0.374-0.556; fine-tuned-context JAAD in-domain AUC was chance-level,
0.493+/-0.021, likely from too little data for the model capacity).

Method: SAME-CLASS mixup (not standard cross-class mixup with soft
labels) -- interpolates pairs of training examples that share the same
true label, keeps the hard 0/1 label on the synthetic example. Chosen
over standard mixup because this project's classification setup uses
hard binary labels throughout (FocalBCELoss, base rate framing) and
because interpolating two DIFFERENT-label crossing trajectories would
synthesize a "trajectory" with no clear physical meaning (a blend of
someone crossing and someone not crossing isn't a coherent hypothetical
pedestrian). Same-class mixup avoids both issues: the synthetic sample
is a plausible variation WITHIN one behavior class, with an
unambiguous label.

Applied at the FEATURE level (after get_data() has produced numeric
tensors for all modalities: local_box, local_context, pose, box), not
the raw-image level -- this project's default best method
(GRL+contrastive) uses precomputed VGG features for local_box/
local_context (continuous vectors, well-defined to interpolate), not
raw pixels, so mixup is well-defined across every modality without
needing to touch the data-generation pipeline. The SAME interpolation
weight alpha is applied consistently across all modalities for one
synthetic sample (they represent one synthetic underlying trajectory,
not four independently-blended things).
"""

import numpy as np


def mixup_same_class(inputs, labels, n_synthetic, alpha=0.2, rng=None, auxiliary=None):
    """inputs: list of per-modality arrays, each shape
    (n_samples, obs_length, feat_dim). labels: array shape (n_samples, 1)
    or (n_samples,), binary. n_synthetic: how many new synthetic samples
    to generate PER CLASS (so 2*n_synthetic total new samples, unless a
    class has fewer than 2 real examples to pair, in which case that
    class is skipped with a logged warning -- can't mixup a class with
    <2 members). alpha: Beta(alpha, alpha) mixing-coefficient
    distribution parameter -- alpha=0.2 (this function's default) is
    the standard mixup paper's recommended value for producing mostly
    near-0-or-near-1 mixing weights (i.e. synthetic samples that lean
    close to one parent or the other, rather than a uniform 50/50 blend
    every time), matching the intuition that most synthetic samples
    should stay close to a real one instead of averaging away
    distinguishing detail. auxiliary: an optional extra array (e.g. a
    contrastive loss's jittered-box view) that must stay index-aligned
    with `inputs`/`labels` -- mixed with the SAME (i, j, lam) triples as
    the main inputs, so a caller that also needs a parallel array
    doesn't have to re-derive which pairs were used.

    Returns (new_inputs, new_labels) normally, or
    (new_inputs, new_labels, new_auxiliary) if `auxiliary` was given.
    inputs/labels/auxiliary are the ORIGINAL arrays with synthetic
    samples APPENDED (not replacing anything), so this is pure
    augmentation -- every real training example is still used exactly
    as before, synthetic ones add to the pool.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    labels_flat = np.asarray(labels).reshape(-1)
    pos_idx = np.where(labels_flat == 1)[0]
    neg_idx = np.where(labels_flat == 0)[0]

    synthetic_inputs = [[] for _ in inputs]
    synthetic_aux = [] if auxiliary is not None else None
    synthetic_labels = []

    for class_idx, class_label in [(pos_idx, 1), (neg_idx, 0)]:
        if len(class_idx) < 2:
            print(f'mixup_same_class: skipping label={class_label}, only '
                 f'{len(class_idx)} real example(s) (need >=2 to pair)')
            continue
        for _ in range(n_synthetic):
            i, j = rng.choice(class_idx, size=2, replace=False)
            lam = rng.beta(alpha, alpha)
            for m, modality_arr in enumerate(inputs):
                arr = np.asarray(modality_arr)
                synthetic_sample = lam * arr[i] + (1 - lam) * arr[j]
                synthetic_inputs[m].append(synthetic_sample)
            if auxiliary is not None:
                aux_arr = np.asarray(auxiliary)
                synthetic_aux.append(lam * aux_arr[i] + (1 - lam) * aux_arr[j])
            synthetic_labels.append(class_label)

    if not synthetic_labels:
        # Neither class had >=2 examples -- nothing to add.
        if auxiliary is not None:
            return inputs, labels, auxiliary
        return inputs, labels

    new_inputs = []
    for m, modality_arr in enumerate(inputs):
        arr = np.asarray(modality_arr)
        synth_arr = np.stack(synthetic_inputs[m], axis=0)
        new_inputs.append(np.concatenate([arr, synth_arr], axis=0))

    synthetic_labels_arr = np.asarray(synthetic_labels).reshape(-1, 1).astype(labels_flat.dtype)
    orig_labels_2d = np.asarray(labels).reshape(-1, 1)
    new_labels = np.concatenate([orig_labels_2d, synthetic_labels_arr], axis=0)

    print(f'mixup_same_class: {len(labels_flat)} real -> {len(new_labels)} total '
         f'(+{len(synthetic_labels)} synthetic, {sum(1 for l in synthetic_labels if l==1)} '
         f'pos / {sum(1 for l in synthetic_labels if l==0)} neg)')

    if auxiliary is not None:
        aux_arr = np.asarray(auxiliary)
        new_aux = np.concatenate([aux_arr, np.stack(synthetic_aux, axis=0)], axis=0)
        return new_inputs, new_labels, new_aux

    return new_inputs, new_labels

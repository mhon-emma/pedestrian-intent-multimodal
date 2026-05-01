"""
Shared utilities for SF-GRU ablation scripts (B, D, and variants).
"""

import os
import sys
import types
import logging
import pickle
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SFGRU_DIR    = '/media/emma/data/MMML/SF-GRU'
PIE_UTIL_DIR = '/media/emma/data/MMML/PIE/utilities'
PIE_DATA_DIR = '/media/emma/data/MMML/PIE_data'
MODEL_PATH   = '/media/emma/data/MMML/SF-GRU/data/models/pie/sf-rnn'
RESULTS_DIR  = '/media/emma/data/MMML/SF-GRU/results'

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Keras compatibility patches (TF 2.14) ────────────────────────────────────
sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
os.chdir(SFGRU_DIR)

import keras
import keras.layers as kl
import keras.regularizers as kr

_rec = types.ModuleType('keras.layers.recurrent')
_rec.GRU = kl.GRU
sys.modules['keras.layers.recurrent'] = _rec

_core = types.ModuleType('keras.layers.core')
_core.regularizers = kr
sys.modules['keras.layers.core'] = _core

# ── Patch get_path to redirect feature cache to PIE_data ─────────────────────
import utils as _sfgru_utils
import sf_gru as _sfgru_mod

_FEAT_DIR = os.path.join(PIE_DATA_DIR, 'features')
_POSE_DIR = os.path.join(SFGRU_DIR, 'data', 'features', 'pie', 'poses')

def _patched_get_path(file_name='', save_folder='models', dataset='pie',
                      save_root_folder='data/'):
    if save_root_folder == 'data/features':
        save_path = (_POSE_DIR if save_folder == 'poses'
                     else os.path.join(_FEAT_DIR, dataset, save_folder))
    else:
        save_path = os.path.join(save_root_folder, dataset, save_folder)
    os.makedirs(save_path, exist_ok=True)
    return os.path.join(save_path, file_name), save_path

_sfgru_utils.get_path = _patched_get_path
_sfgru_mod.get_path   = _patched_get_path

# ── Colour scheme ─────────────────────────────────────────────────────────────
MODALITY_COLORS = {
    'local_box':     '#4C72B0',
    'local_context': '#DD8452',
    'pose':          '#55A868',
    'box':           '#C44E52',
    'speed':         '#8172B2',
}

# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s  %(levelname)s  %(message)s',
                                          datefmt='%H:%M:%S'))
        log.addHandler(ch)
    return log

# ── Data / model loading ──────────────────────────────────────────────────────
_DATA_OPTS = {
    'fstride': 1, 'subset': 'default', 'data_split_type': 'default',
    'seq_type': 'crossing', 'min_track_size': 75,
}
_MODEL_OPTS = {
    'obs_input_type': ['local_box', 'local_context', 'pose', 'box', 'speed'],
    'enlarge_ratio': 1.5, 'pred_target_type': ['crossing'],
    'obs_length': 15, 'time_to_event': 60,
    'dataset': 'pie', 'normalize_boxes': True,
}

def load_test_data():
    """Returns (inputs, labels, data_types)."""
    from pie_data import PIE
    from sf_gru import SFGRU
    imdb   = PIE(data_path=PIE_DATA_DIR)
    beh    = imdb.generate_data_trajectory_sequence('test', **_DATA_OPTS)
    method = SFGRU()
    td, data_types, _ = method.get_data({'test': beh}, _MODEL_OPTS)
    return td['test'][0], td['test'][1].astype(int), data_types

def load_model():
    from keras.models import load_model as _load
    return _load(os.path.join(MODEL_PATH, 'model.h5'))

# ── Metrics ───────────────────────────────────────────────────────────────────
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)

def compute_metrics(labels, preds, threshold=0.5):
    probs  = preds.flatten()
    binary = (probs >= threshold).astype(int)
    return {
        'acc':  accuracy_score(labels, binary),
        'f1':   f1_score(labels, binary, zero_division=0),
        'prec': precision_score(labels, binary, zero_division=0),
        'rec':  recall_score(labels, binary, zero_division=0),
        'auc':  roc_auc_score(labels, probs),
    }

def fmt_metrics(m):
    return (f"acc={m['acc']:.4f}  f1={m['f1']:.4f}  "
            f"prec={m['prec']:.4f}  rec={m['rec']:.4f}  auc={m['auc']:.4f}")

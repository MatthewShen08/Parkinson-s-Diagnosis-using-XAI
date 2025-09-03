#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parkinson's voice screening: MLP + CNN + BiGRU with MKL-style fusion
- Inputs:
    (A) Tabular prosodic features (Praat + librosa stats)  -> MLP branch
    (B) Log-mel spectrogram (n_mels x T)                   -> 2D-CNN branch
    (C) MFCC sequence (T x n_mfcc)                         -> BiGRU branch

- Fusion: MKL-inspired layer that learns non-negative convex weights over
  multiple Gaussian kernels per branch and across branches (softmax constraints).

- Evaluation: subject-aware nested CV (outer StratifiedGroupKFold; inner GroupKFold).
  Picks a fixed-sensitivity threshold (default 0.90) on inner validation after
  temperature scaling. Reports ROC-AUC, PR-AUC, Acc, Prec, Rec, F1 with 95% CIs.
  Saves fold artifacts (weights, scaler, temperature, threshold).

- Final fit: trains on all labeled data, calibrates a single temperature using
  5-fold OOF logits, sets operating threshold for target sensitivity on OOF, then
  scores any blind folder.

Usage
-----
python pd_voice_mkl.py \
  --hc_dir "C:/Users/mtshe/Downloads/HC_AH" \
  --pd_dir "C:/Users/mtshe/Downloads/PD_AH" \
  --blind_dir "C:/Users/mtshe/Downloads/Blind Data of 81" \
  --out_dir "./pd_out" \
  --target_sensitivity 0.90
"""

import os, sys, time, json, math, argparse, logging, platform, warnings
from typing import Tuple, List, Dict, Optional

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", None)

import librosa
import parselmouth
from parselmouth.praat import call

import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix,
                             precision_recall_curve, roc_curve)
from sklearn.utils import resample

import tensorflow as tf
from tensorflow.keras import layers as L, models as M, callbacks as C, initializers as I, regularizers as R


SR = 16000                 # resample rate
N_FFT = 1024
HOP = 256
N_MELS = 64
N_MFCC = 20
T_FRAMES = 128            # fixed time frames after pad/crop

TABULAR_FEATURES = [
    # Will populate dynamically (Praat + librosa stats + MFCC mean/std)
]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

SCORE_BINS = [
    (0.00, 0.10, "Very low likelihood of PD"),
    (0.10, 0.20, "Low likelihood of PD"),
    (0.20, 0.30, "Mild likelihood of PD"),
    (0.30, 0.40, "Moderate likelihood of PD"),
    (0.40, 0.50, "Moderate to high likelihood of PD"),
    (0.50, 0.60, "High likelihood of PD"),
    (0.60, 0.70, "Very high likelihood of PD"),
    (0.70, 0.80, "Extremely high likelihood of PD"),
    (0.80, 0.90, "Near certainty of PD"),
    (0.90, 1.01, "Definite likelihood of PD"),
]


def setup_logging(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_path, mode="w", encoding="utf-8")]
    )
    logging.info("Output: %s", out_dir)
    logging.info("Python %s | TF %s | Platform %s",
                 sys.version.split()[0], tf.__version__, platform.platform())


def ensure_dir(d: str):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def bootstrap_ci(values: np.ndarray, stat_fn, n_boot=2000, alpha=0.05, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    vals = np.asarray(values, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan)
    boots = []
    n = len(vals)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        boots.append(stat_fn(vals[idx]))
    lo = np.percentile(boots, 100*alpha/2)
    hi = np.percentile(boots, 100*(1 - alpha/2))
    return float(lo), float(hi)

def parse_subject_id(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    if '-' in stem:
        base = stem.split('-')[0]
    else:
        base = stem
    if '_' in base:
        parts = base.split('_')
        if len(parts) > 2:
            base = "_".join(parts[:2])
    return base

def pad_or_crop(x: np.ndarray, target_len: int, axis: int) -> np.ndarray:
    # Pad or crop along 'axis' to 'target_len'
    cur = x.shape[axis]
    if cur == target_len:
        return x
    if cur > target_len:
        slc = [slice(None)] * x.ndim
        slc[axis] = slice(0, target_len)
        return x[tuple(slc)]
    pad_width = [(0,0)] * x.ndim
    pad_width[axis] = (0, target_len - cur)
    return np.pad(x, pad_width, mode="constant")

def score_band(p: float) -> str:
    for lo, hi, desc in SCORE_BINS:
        if lo <= p < hi:
            return desc
    return "Unscored"


def _pitch_stats(sound: parselmouth.Sound) -> Dict[str, float]:
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[np.isfinite(f0)]
    f0 = f0[f0 > 0]
    if f0.size == 0:
        return dict(mean_pitch=np.nan, median_pitch=np.nan, std_pitch=np.nan, p25_pitch=np.nan, p75_pitch=np.nan)
    return dict(
        mean_pitch=float(np.mean(f0)),
        median_pitch=float(np.median(f0)),
        std_pitch=float(np.std(f0)),
        p25_pitch=float(np.percentile(f0, 25)),
        p75_pitch=float(np.percentile(f0, 75)),
    )

def extract_all_inputs(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Returns:
      spec: (N_MELS, T_FRAMES, 1)   log-mel spectrogram
      mfcc_seq: (T_FRAMES, N_MFCC)  MFCC sequence
      tab: (D,)                     tabular features
      meta: dict                    extra meta if needed
    """

    snd = parselmouth.Sound(path)
    f0s = _pitch_stats(snd)
    pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    local_jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    local_shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harm = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    mean_hnr = call(harm, "Get mean", 0, 0)
    duration = snd.get_total_duration()


    y, sr = librosa.load(path, sr=SR, mono=True)
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    if y_trim.size == 0:
        y_trim = y

    # Log-mel spectrogram
    S = librosa.feature.melspectrogram(y=y_trim, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    S_db = librosa.power_to_db(S, ref=1.0)
    S_db = pad_or_crop(S_db, T_FRAMES, axis=1)
    spec = S_db[..., None].astype(np.float32)  # (mels, T, 1)

    # MFCC sequence (T x n_mfcc)
    MF = librosa.feature.mfcc(S=librosa.power_to_db(S), sr=sr, n_mfcc=N_MFCC)
    MF = pad_or_crop(MF, T_FRAMES, axis=1)
    mfcc_seq = MF.T.astype(np.float32)  # (T, n_mfcc)

    # Additional stats for tabular
    centroid = librosa.feature.spectral_centroid(y=y_trim, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y_trim, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y_trim)
    rms = librosa.feature.rms(y=y_trim)

    tab = {
        **f0s,
        "local_jitter": float(local_jitter),
        "local_shimmer": float(local_shimmer),
        "mean_hnr": float(mean_hnr),
        "duration": float(duration),
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "rolloff_mean": float(np.mean(rolloff)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
    }

    # MFCC mean/std (augment tabular)
    tab.update({f"mfcc_{i+1}_mean": float(np.mean(MF[i])) for i in range(N_MFCC)})
    tab.update({f"mfcc_{i+1}_std":  float(np.std(MF[i]))  for i in range(N_MFCC)})

    return spec.astype(np.float32), mfcc_seq.astype(np.float32), np.array(list(tab.values()), dtype=np.float32), tab

def build_dataset(root: str, label: Optional[int]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int], List[str], List[str], List[Dict[str,float]]]:
    specs, mfccs, tabs, labels, files, subjects, tabs_dict = [], [], [], [], [], [], []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".wav"):
                fp = os.path.join(r, f)
                try:
                    spec, seq, tab_arr, tab_map = extract_all_inputs(fp)
                    specs.append(spec)
                    mfccs.append(seq)
                    tabs.append(tab_arr)
                    labels.append(label if label is not None else -1)
                    files.append(fp)
                    subjects.append(parse_subject_id(f))
                    tabs_dict.append(tab_map)
                except Exception as e:
                    logging.warning("Failed on %s: %s", fp, e)
    return specs, mfccs, tabs, labels, files, subjects, tabs_dict


class MKLFusion(L.Layer):
    """
    MKL-inspired fusion:
      - Projects each branch embedding to a common latent dim d.
      - For each branch, K Gaussian kernels with trainable centers and bandwidths.
      - Softmax over kernels per branch (non-negative, sum to 1).
      - Softmax over branches to mix branch-level kernel scores.
      - Returns: fused representation (concat weighted projections) + scalar kernel score.

    This yields non-negative, convex mixtures in spirit of MKL while remaining end-to-end trainable.
    """
    def __init__(self, latent_dim=64, kernels_per_branch=3, name="mkl_fusion"):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.kpb = kernels_per_branch

    def build(self, input_shapes):
        # One projection per branch
        self.proj_layers = [L.Dense(self.latent_dim, activation=None, use_bias=True) for _ in input_shapes]
        # Per-branch kernels: centers and log_sigmas
        self.centers = []
        self.log_sigmas = []
        self.kernel_logits = []   # per-branch kernel weights (softmaxed)
        for _ in input_shapes:
            self.centers.append(self.add_weight(shape=(self.kpb, self.latent_dim),
                                               initializer=I.GlorotUniform(), trainable=True, name="centers"))
            self.log_sigmas.append(self.add_weight(shape=(self.kpb,),
                                                   initializer=I.Zeros(), trainable=True, name="log_sigmas"))
            self.kernel_logits.append(self.add_weight(shape=(self.kpb,),
                                                      initializer=I.Zeros(), trainable=True, name="kernel_logits"))
        # Branch mixture logits
        self.branch_logits = self.add_weight(shape=(len(input_shapes),),
                                             initializer=I.Zeros(), trainable=True, name="branch_logits")

    def call(self, inputs, training=None):
        # inputs: list of [B, d_i] embeddings
        proj = [L.Activation("relu")(p(x)) for p, x in zip(self.proj_layers, inputs)]  # list of [B, d]
        B = tf.shape(proj[0])[0]
        k_scores = []  # per-branch kernel score [B, K] -> reduce to [B,1]
        for i, h in enumerate(proj):
            # [B,1,d] - [1,K,d] -> [B,K,d]
            h_exp = tf.expand_dims(h, 1)
            c = self.centers[i]          # [K,d]
            log_s = self.log_sigmas[i]   # [K]
            diff = h_exp - c             # [B,K,d]
            dist2 = tf.reduce_sum(diff*diff, axis=-1)  # [B,K]
            inv_two_sigma2 = tf.exp(-2.0*log_s) * 0.5  # since sigma = exp(log_s)
            k = tf.exp(-dist2 * inv_two_sigma2)        # Gaussian kernels [B,K]
            w = tf.nn.softmax(self.kernel_logits[i])   # [K]
            k_scores.append(tf.reduce_sum(k * w, axis=1, keepdims=True))  # [B,1]

        # Branch weighting
        a = tf.nn.softmax(self.branch_logits)  # [branches]
        branch_score = tf.add_n([a[i]*k_scores[i] for i in range(len(proj))])  # [B,1]

        # Fused representation: weighted concat of projected branches
        fused = tf.concat([a[i]*proj[i] for i in range(len(proj))], axis=-1)  # [B, branches*latent]
        return fused, branch_score  # return both; caller decides how to use

def build_model(tab_dim: int) -> M.Model:
    # Branch A: Tabular (MLP)
    inp_tab = L.Input(shape=(tab_dim,), name="tabular")
    xA = L.Dense(64, activation="relu")(inp_tab)
    xA = L.Dropout(0.2)(xA)
    xA = L.Dense(64, activation="relu")(xA)

    # Branch B: Spectrogram (CNN 2D)
    inp_spec = L.Input(shape=(N_MELS, T_FRAMES, 1), name="spec")
    xB = L.Conv2D(32, (3,3), padding="same", activation="relu")(inp_spec)
    xB = L.MaxPooling2D((2,2))(xB)
    xB = L.Conv2D(64, (3,3), padding="same", activation="relu")(xB)
    xB = L.MaxPooling2D((2,2))(xB)
    xB = L.Conv2D(128, (3,3), padding="same", activation="relu")(xB)
    xB = L.GlobalAveragePooling2D()(xB)
    xB = L.Dropout(0.3)(xB)
    xB = L.Dense(64, activation="relu")(xB)

    # Branch C: MFCC sequence (BiGRU)
    inp_seq = L.Input(shape=(T_FRAMES, N_MFCC), name="mfcc_seq")
    xC = L.Bidirectional(L.GRU(64, return_sequences=True))(inp_seq)
    xC = L.Bidirectional(L.GRU(32, return_sequences=False))(xC)
    xC = L.Dropout(0.3)(xC)
    xC = L.Dense(64, activation="relu")(xC)

    # MKL-inspired fusion
    fused, kscore = MKLFusion(latent_dim=64, kernels_per_branch=3)([xA, xB, xC])  # fused: [B, 3*64], kscore: [B,1]

    # Final head
    z = L.Concatenate()([fused, kscore])
    z = L.Dense(64, activation="relu")(z)
    z = L.Dropout(0.3)(z)
    logit = L.Dense(1, activation=None, name="logit")(z)
    prob = L.Activation("sigmoid", name="prob")(logit)

    model = M.Model(inputs=[inp_tab, inp_spec, inp_seq], outputs=[prob, logit], name="PD_MLP_CNN_GRU_MKL")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  loss={"prob": "binary_crossentropy", "logit": None})
    return model


class TemperatureScaler(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # T > 0; optimize logT for stability
        self.logT = tf.Variable(0.0, trainable=True, dtype=tf.float32)

    @tf.function
    def call(self, logits):
        T = tf.math.softplus(self.logT) + 1e-6
        return logits / T

def optimize_temperature(val_logits: np.ndarray, y_val: np.ndarray, steps=500, lr=0.05, verbose=False) -> float:
    y = tf.constant(y_val.reshape(-1,1), dtype=tf.float32)
    logits = tf.constant(val_logits.reshape(-1,1), dtype=tf.float32)
    scaler = TemperatureScaler()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            scaled = scaler(logits)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=scaled))
        grads = tape.gradient(loss, [scaler.logT])
        opt.apply_gradients(zip(grads, [scaler.logT]))
    T = float(tf.math.softplus(scaler.logT).numpy() + 1e-6)
    if verbose:
        print("Optimized temperature:", T)
    return T

def choose_threshold_fixed_sens(y_true: np.ndarray, probs: np.ndarray, target_sens: float) -> float:
    # Find highest threshold achieving at least target sensitivity (recall on positives)
    fpr, tpr, thr = roc_curve(y_true, probs)
    # tpr = sensitivity
    candidates = thr[tpr >= target_sens]
    if len(candidates) == 0:
        # fallback: Youden J
        j = tpr - fpr
        return float(thr[np.argmax(j)])
    return float(np.max(candidates))

def compute_metrics(y_true: np.ndarray, probs: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (probs >= thr).astype(int)
    out = dict(
        roc_auc = roc_auc_score(y_true, probs) if len(np.unique(y_true))>1 else np.nan,
        pr_auc  = average_precision_score(y_true, probs) if len(np.unique(y_true))>1 else np.nan,
        acc     = accuracy_score(y_true, y_pred),
        prec    = precision_score(y_true, y_pred, zero_division=0),
        rec     = recall_score(y_true, y_pred, zero_division=0),
        f1      = f1_score(y_true, y_pred, zero_division=0),
    )
    return out

def plot_reliability(y_true, probs, out_png):
    # Reliability diagram
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot([0,1], [0,1], "--")
    plt.plot(prob_pred, prob_true, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_roc_pr(y_true, probs, out_prefix):
    fpr, tpr, _ = roc_curve(y_true, probs)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    # ROC
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC")
    plt.tight_layout(); plt.savefig(out_prefix+"_roc.png", dpi=200); plt.close()
    # PR
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.tight_layout(); plt.savefig(out_prefix+"_pr.png", dpi=200); plt.close()

def load_labeled(hc_dir: str, pd_dir: str):
    sA, mA, tA, yA, fA, gA, mapA = build_dataset(hc_dir, label=0)
    sB, mB, tB, yB, fB, gB, mapB = build_dataset(pd_dir, label=1)
    specs = np.stack(sA + sB, axis=0)
    mfccs = np.stack(mA + mB, axis=0)
    tabs  = np.stack(tA + tB, axis=0)
    y     = np.array(yA + yB, dtype=int)
    files = fA + fB
    groups= np.array(gA + gB)
    # Store feature names
    feat_names = list(mapA[0].keys()) if len(mapA)>0 else list(mapB[0].keys())
    return specs, mfccs, tabs, y, files, groups, feat_names

def load_blind(blind_dir: str, feat_names: List[str]):
    s, m, t, y, f, g, maps = build_dataset(blind_dir, label=None)
    specs = np.stack(s, axis=0)
    mfccs = np.stack(m, axis=0)
    # Align tabular columns order
    cols = feat_names
    t_df = pd.DataFrame(maps)
    t_df = t_df.reindex(columns=cols)
    tabs = t_df.values.astype(np.float32)
    return specs, mfccs, tabs, f, g

def run_nested_cv(specs, mfccs, tabs, y, groups, out_dir, target_sens=0.90, n_splits=5):
    ensure_dir(out_dir)
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    fold_rows = []
    all_test_probs = np.zeros_like(y, dtype=float)

    for fold, (tr_idx, te_idx) in enumerate(outer.split(tabs, y, groups), 1):
        logging.info("Fold %d/%d | Train=%d | Test=%d", fold, n_splits, len(tr_idx), len(te_idx))

        # Inner split for early stopping / temperature / threshold
        inner = GroupKFold(n_splits=4)
        # Choose the first inner split
        tr_in_idx, va_in_idx = next(inner.split(tabs[tr_idx], y[tr_idx], groups[tr_idx]))
        tr_idx2 = tr_idx[tr_in_idx]
        va_idx2 = tr_idx[va_in_idx]

        # Scaler for tabular (fit on inner-train)
        scaler = StandardScaler()
        tabs_tr = scaler.fit_transform(tabs[tr_idx2])
        tabs_va = scaler.transform(tabs[va_idx2])
        tabs_te = scaler.transform(tabs[te_idx])

        # Build model
        model = build_model(tab_dim=tabs_tr.shape[1])

        # Training data
        x_tr = {"tabular": tabs_tr, "spec": specs[tr_idx2], "mfcc_seq": mfccs[tr_idx2]}
        x_va = {"tabular": tabs_va, "spec": specs[va_idx2], "mfcc_seq": mfccs[va_idx2]}
        y_tr = y[tr_idx2]; y_va = y[va_idx2]

        es = C.EarlyStopping(monitor="val_prob_loss", patience=10, restore_best_weights=True)
        rl = C.ReduceLROnPlateau(monitor="val_prob_loss", factor=0.5, patience=5, min_lr=1e-5)
        hist = model.fit(x_tr, {"prob": y_tr, "logit": y_tr},
                         validation_data=(x_va, {"prob": y_va, "logit": y_va}),
                         batch_size=16, epochs=200, callbacks=[es, rl], verbose=0)

        # Inner validation: logits -> temperature -> threshold
        va_probs, va_logits = model.predict(x_va, batch_size=64, verbose=0)
        va_probs = va_probs.reshape(-1)
        va_logits = va_logits.reshape(-1)
        T = optimize_temperature(va_logits, y_va, steps=400, lr=0.05)
        va_probs_cal = 1.0 / (1.0 + np.exp(-va_logits / T))
        thr = choose_threshold_fixed_sens(y_va, va_probs_cal, target_sens=target_sens)

        # Test evaluation
        x_te = {"tabular": tabs_te, "spec": specs[te_idx], "mfcc_seq": mfccs[te_idx]}
        te_probs, te_logits = model.predict(x_te, batch_size=64, verbose=0)
        te_probs = te_probs.reshape(-1)
        te_logits = te_logits.reshape(-1)
        te_probs_cal = 1.0 / (1.0 + np.exp(-te_logits / T))

        all_test_probs[te_idx] = te_probs_cal
        m = compute_metrics(y[te_idx], te_probs_cal, thr)
        cm = confusion_matrix(y[te_idx], (te_probs_cal >= thr).astype(int)).tolist()

        # Save fold artifacts
        fdir = os.path.join(out_dir, f"fold_{fold}")
        ensure_dir(fdir)
        model.save(os.path.join(fdir, "model.keras"))
        joblib.dump(scaler, os.path.join(fdir, "scaler.joblib"))
        with open(os.path.join(fdir, "operating_point.json"), "w") as f:
            json.dump({"temperature": float(T), "threshold": float(thr)}, f, indent=2)
        with open(os.path.join(fdir, "metrics.json"), "w") as f:
            json.dump({**m, "confusion_matrix": cm}, f, indent=2)

        fold_rows.append({"fold": fold, **m, "threshold": thr, "temperature": T})

    # CV summary
    df = pd.DataFrame(fold_rows)
    df.to_csv(os.path.join(out_dir, "cv_metrics.csv"), index=False)
    summary = {}
    for k in ["roc_auc","pr_auc","acc","prec","rec","f1"]:
        arr = df[k].astype(float).values
        mean = float(np.nanmean(arr))
        lo, hi = bootstrap_ci(arr, np.mean)
        summary[k] = {"mean": mean, "ci95":[lo,hi]}
    with open(os.path.join(out_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return df, all_test_probs
    
def fit_full_and_calibrate(specs, mfccs, tabs, y, out_dir, target_sens):
    scaler = StandardScaler()
    tabs_sc = scaler.fit_transform(tabs)

    model = build_model(tab_dim=tabs_sc.shape[1])

    # 5-fold OOF for temperature + threshold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_logits = np.zeros_like(y, dtype=float)

    for tr, va in skf.split(tabs_sc, y):
        x_tr = {"tabular": tabs_sc[tr], "spec": specs[tr], "mfcc_seq": mfccs[tr]}
        x_va = {"tabular": tabs_sc[va], "spec": specs[va], "mfcc_seq": mfccs[va]}
        y_tr, y_va = y[tr], y[va]

        es = C.EarlyStopping(monitor="val_prob_loss", patience=10, restore_best_weights=True)
        rl = C.ReduceLROnPlateau(monitor="val_prob_loss", factor=0.5, patience=5, min_lr=1e-5)
        model.fit(x_tr, {"prob": y_tr, "logit": y_tr},
                  validation_data=(x_va, {"prob": y_va, "logit": y_va}),
                  batch_size=16, epochs=200, callbacks=[es, rl], verbose=0)
        _, logits = model.predict(x_va, batch_size=64, verbose=0)
        oof_logits[va] = logits.reshape(-1)

    T = optimize_temperature(oof_logits, y, steps=500, lr=0.05)
    oof_probs_cal = 1.0 / (1.0 + np.exp(-oof_logits / T))
    thr = choose_threshold_fixed_sens(y, oof_probs_cal, target_sens)

    # Refit on all data
    es = C.EarlyStopping(monitor="val_prob_loss", patience=10, restore_best_weights=True)
    rl = C.ReduceLROnPlateau(monitor="val_prob_loss", factor=0.5, patience=5, min_lr=1e-5)
    # reserve tiny val split just for early stopping stability
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    split = int(0.9*len(y))
    tr, va = idx[:split], idx[split:]

    x_tr = {"tabular": tabs_sc[tr], "spec": specs[tr], "mfcc_seq": mfccs[tr]}
    x_va = {"tabular": tabs_sc[va], "spec": specs[va], "mfcc_seq": mfccs[va]}
    y_tr, y_va = y[tr], y[va]

    model.fit(x_tr, {"prob": y_tr, "logit": y_tr},
              validation_data=(x_va, {"prob": y_va, "logit": y_va}),
              batch_size=16, epochs=200, callbacks=[es, rl], verbose=0)

    # Save artifacts
    model.save(os.path.join(out_dir, "final_model.keras"))
    joblib.dump(scaler, os.path.join(out_dir, "final_scaler.joblib"))
    with open(os.path.join(out_dir, "operating_point_final.json"), "w") as f:
        json.dump({"temperature": float(T), "threshold": float(thr)}, f, indent=2)

    # SHAP on a small background to keep memory sane
    # Use the 'prob' head's pre-activation (logit) path's penultimate layer input: we already have "logit"
    # For simplicity, explain tabular branch (tree explainer not applicable; we use KernelExplainer on prob output)
    # We’ll provide a lightweight summary for top tabular features.
    try:
        bg_idx = np.random.choice(len(y), size=min(50, len(y)), replace=False)
        explainer = shap.KernelExplainer(
            lambda X: model.predict({"tabular": X.astype(np.float32),
                                     "spec": specs[bg_idx],
                                     "mfcc_seq": mfccs[bg_idx]}, verbose=0)[0],
            tabs_sc[bg_idx]
        )
        shap_vals = explainer.shap_values(tabs_sc[bg_idx], nsamples=100)
        shap.summary_plot(shap_vals, pd.DataFrame(tabs_sc[bg_idx]), show=False, max_display=20)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "shap_tabular_top20.png"), dpi=300); plt.close()
    except Exception as e:
        logging.warning("SHAP summary skipped: %s", e)

    return model, scaler, T, thr


def score_blind(model, scaler, T, thr, specs, mfccs, tabs, files, subjects, out_dir):
    tabs_sc = scaler.transform(tabs)
    x = {"tabular": tabs_sc, "spec": specs, "mfcc_seq": mfccs}
    probs, logits = model.predict(x, batch_size=64, verbose=0)
    logits = logits.reshape(-1)
    probs_cal = 1.0 / (1.0 + np.exp(-logits / T))
    pred = (probs_cal >= thr).astype(int)
    bands = [score_band(p) for p in probs_cal]
    out = pd.DataFrame({"file": files, "subject_id": subjects, "prob_pd": probs_cal,
                        "pred_label": pred, "score_band": bands}).sort_values("prob_pd", ascending=False)
    out.to_csv(os.path.join(out_dir, "blind_predictions.csv"), index=False)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hc_dir", required=True, type=str)
    ap.add_argument("--pd_dir", required=True, type=str)
    ap.add_argument("--blind_dir", default=None, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--target_sensitivity", type=float, default=0.90)
    args = ap.parse_args()

    setup_logging(args.out_dir)
    t0 = time.time()

    logging.info("Loading labeled data…")
    specs, mfccs, tabs, y, files, groups, feat_names = load_labeled(args.hc_dir, args.pd_dir)
    with open(os.path.join(args.out_dir, "feature_names.json"), "w") as f:
        json.dump(feat_names, f, indent=2)

    logging.info("Nested CV (%d folds)…", 5)
    df_cv, test_probs = run_nested_cv(specs, mfccs, tabs, y, groups, args.out_dir, target_sens=args.target_sensitivity, n_splits=5)

    # Aggregate ROC/PR plots from OOF
    plot_roc_pr(y, test_probs, os.path.join(args.out_dir, "cv_oof"))
    plot_reliability(y, test_probs, os.path.join(args.out_dir, "cv_calibration.png"))

    logging.info("Fitting final model and calibrating…")
    model, scaler, T, thr = fit_full_and_calibrate(specs, mfccs, tabs, y, args.out_dir, args.target_sensitivity)

    if args.blind_dir:
        logging.info("Scoring blind folder…")
        sB, mB, tB, fB, gB = load_blind(args.blind_dir, feat_names)
        out = score_blind(model, scaler, T, thr, sB, mB, tB, fB, gB, args.out_dir)
        # Print top few for quick glance
        logging.info("\n%s", out.head(10).to_string(index=False))

    dt = time.time() - t0
    logging.info("Done in %.1fs. Artifacts in %s", dt, args.out_dir)

if __name__ == "__main__":
    main()

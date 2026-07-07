"""
cell_classifier_v5.py
─────────────────────
Classification multi-canal de cellules rénales de rat (AQP2/AE1).

Détection des noyaux (canal blanc) puis classification par échantillonnage
radial des canaux vert (AQP2), rouge (AE1) et bleu.

Améliorations v5 :
  - Architecture modulaire : main(), fonctions pures, point d'entrée
  - Configuration : dataclass Config + argparse (CLI)
  - Typage statique complet (type hints)
  - Barres de progression tqdm
  - build_bg_map vectorisé (reshape au lieu de boucles Python)
  - Gestion d'erreurs robuste (try/except, messages explicites)
  - Docstrings complètes
  - Pré-calcul des coordonnées de patch pour éviter les meshgrid répétés
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
from czifile import imread
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Paramètres de détection et de classification."""

    # Fichier
    czi_path: str = r"C:\Users\pimfa\Documents\MAIA\Rein_de_rat\transfer_12956536_files_7b6ebfa1\2021_05_18__0986gtAQP2nov_rbAE1.czi"

    # Canaux
    ch_white: int = 0
    ch_green: int = 1
    ch_red: int = 2
    ch_blue: int = 3

    # Redimensionnement maximal (None = pas de downscale)
    max_dim: Optional[int] = 8192

    # Normalisation robuste (clip percentile) avant conversion en uint8.
    # Remplace le min/max brut : quelques pixels chauds (saturation capteur,
    # poussière, artefact ponctuel) ne doivent pas à eux seuls écraser tout
    # le reste du signal dans les valeurs basses. norm_clip_low/high sont
    # les percentiles (0-100) utilisés comme bornes noir/blanc.
    norm_clip_low: float = 0.5
    norm_clip_high: float = 99.5

    # Détection noyaux
    min_area: int = 50
    max_area: int = 80000
    merge_ratio: float = 1.6

    # Géométrie d'échantillonnage
    nucleus_r: int = 12      # rayon du noyau (disque central)
    halo_r: int = 45        # rayon externe de l'anneau de recherche

    # GREEN – anneau membranaire (AQP2 apical)
    green_thresh: int = 30
    green_bg_ratio: float = 2.0

    # RED – max(disque, anneau) (AE1 basolatéral)
    red_thresh: int = 50
    red_stat: str = "p95"      # "mean", "p95", "max" – p95 capture le signal punctiforme
    red_bg_ratio: float = 1.2
    red_not_greener: bool = False

    # BLUE – max(disque, anneau)
    blue_thresh: int = 20
    blue_bg_ratio: float = 1.5

    # Fond local
    bg_floor_frac: float = 0.15   # plancher = fraction × médiane globale
    bg_grid: int = 8               # grille de division pour la carte de fond

    # Couleurs d'annotation (BGR)
    c_green: Tuple[int, int, int] = (0, 255, 0)
    c_red: Tuple[int, int, int] = (0, 0, 255)
    c_blue: Tuple[int, int, int] = (255, 80, 0)
    c_gr: Tuple[int, int, int] = (0, 255, 255)
    c_gb: Tuple[int, int, int] = (255, 255, 0)
    c_rb: Tuple[int, int, int] = (255, 0, 255)
    c_grb: Tuple[int, int, int] = (255, 255, 255)
    c_unc: Tuple[int, int, int] = (80, 80, 80)

    @property
    def channels(self) -> Tuple[int, int, int, int]:
        return (self.ch_white, self.ch_green, self.ch_red, self.ch_blue)


def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI surchargeant la configuration."""
    p = argparse.ArgumentParser(description="Classification cellulaire multi-canal (AQP2/AE1)")
    p.add_argument("czi_path", nargs="?", help="Chemin vers le fichier CZI")
    p.add_argument("--max-dim", type=int, default=8192, help="Dimension max après redimensionnement")
    p.add_argument("--clip-low", type=float, default=0.5,
                   help="Percentile bas (0-100) pour la normalisation robuste des canaux")
    p.add_argument("--clip-high", type=float, default=99.5,
                   help="Percentile haut (0-100) pour la normalisation robuste des canaux")
    p.add_argument("--min-area", type=int, default=50, help="Aire minimale d'un blob")
    p.add_argument("--max-area", type=int, default=80000, help="Aire maximale d'un blob")
    p.add_argument("--nucleus-r", type=int, default=12, help="Rayon du noyau (px)")
    p.add_argument("--halo-r", type=int, default=38, help="Rayon du halo (px)")
    p.add_argument("--green-thresh", type=int, default=30, help="Seuil signal vert")
    p.add_argument("--green-ratio", type=float, default=2.0, help="Rapport vert / fond")
    p.add_argument("--red-thresh", type=int, default=50, help="Seuil signal rouge (p95)")
    p.add_argument("--red-stat", default="p95", help="Statistique rouge: mean, p95, max")
    p.add_argument("--red-ratio", type=float, default=1.4, help="Rapport rouge / fond")
    p.add_argument("--blue-thresh", type=int, default=20, help="Seuil signal bleu")
    p.add_argument("--blue-ratio", type=float, default=1.5, help="Rapport bleu / fond")
    p.add_argument("--bg-grid", type=int, default=8, help="Grille pour la carte de fond")
    p.add_argument("--self-train", action="store_true",
                   help="Auto-entraînement : le RF apprend des prédictions threshold et les corrige")
    p.add_argument("--model-path", default="cell_classifier_model.pkl",
                   help="Chemin du modèle (sauvé si --self-train, chargé s'il existe)")
    p.add_argument("--train", metavar="CSV", help=argparse.SUPPRESS)
    return p


ParsedArgs = argparse.Namespace


def config_from_args(args: Optional[List[str]] = None) -> Tuple[Config, ParsedArgs]:
    """Crée un Config et retourne le namespace argparse complet."""
    cfg = Config()
    parser = build_parser()
    parsed = parser.parse_args(args)
    if parsed.czi_path:
        cfg.czi_path = parsed.czi_path
    for field_name in ("max_dim", "clip_low", "clip_high", "min_area", "max_area", "nucleus_r", "halo_r",
                       "green_thresh", "green_ratio", "red_thresh", "red_stat",
                       "red_ratio", "blue_thresh", "blue_ratio", "bg_grid"):
        val = getattr(parsed, field_name, None)
        if val is not None:
            attr_map = {
                "green_ratio": "green_bg_ratio",
                "red_ratio": "red_bg_ratio",
                "blue_ratio": "blue_bg_ratio",
                "clip_low": "norm_clip_low",
                "clip_high": "norm_clip_high",
            }
            setattr(cfg, attr_map.get(field_name, field_name), val)
    return cfg, parsed


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITAIRES IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def to_uint8(
    arr: np.ndarray,
    max_dim: Optional[int] = 8192,
    clip_low: float = 0.5,
    clip_high: float = 99.5,
) -> np.ndarray:
    """Normalise un canal en uint8 [0, 255] après downscale optionnel.

    Normalisation ROBUSTE par clip percentile plutôt que min/max brut.
    Avec un min/max classique, un seul pixel chaud (saturation capteur,
    poussière fluorescente, artefact ponctuel) fixe la borne haute et
    écrase tout le signal réel dans les valeurs basses -> faux négatifs
    en aval (seuils GREEN/RED/BLUE jamais atteints). En clippant aux
    percentiles [clip_low, clip_high] avant de ramener sur [0, 255], ces
    quelques pixels extrêmes sont saturés au lieu de dicter toute
    l'échelle, et le contraste utile est préservé.

    clip_low=0/clip_high=100 redonne exactement l'ancien comportement
    min/max.
    """
    if max_dim is not None:
        h, w = arr.shape[:2]
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            arr = cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))),
                             interpolation=cv2.INTER_AREA)
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [clip_low, clip_high])
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    else:
        # Image quasi constante (percentiles égaux) : repli sur min/max
        # brut pour éviter une division par zéro.
        mn, mx = arr.min(), arr.max()
        arr = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)
    return (arr * 255).astype(np.uint8)


def build_bg_map(ch_u8: np.ndarray, grid: int = 8) -> np.ndarray:
    """Carte de fond local : percentile 25 par tuile, interpolée.

    Version vectorisée sans boucle Python explicite sur les tuiles.
    """
    H, W = ch_u8.shape
    th = max(1, H // grid)
    tw = max(1, W // grid)
    # Découpage en grid×grid blocs sans boucle
    H_trim = th * grid
    W_trim = tw * grid
    trimmed = ch_u8[:H_trim, :W_trim]
    # Reshape en (grid, th, grid, tw) → percentile sur axes 1 et 3
    blocks = trimmed.reshape(grid, th, grid, tw)
    # Percentile 25 sur chaque bloc
    bg_low = np.percentile(blocks, 25, axis=(1, 3), keepdims=False).astype(np.float32)
    return cv2.resize(bg_low, (W, H), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
#  ÉCHANTILLONNAGE RADIAL
# ─────────────────────────────────────────────────────────────────────────────

def _make_ring_coords(
    shape: Tuple[int, int], cx: int, cy: int, r_inner: int, r_outer: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coordonnées locales et masque booléen pour un anneau.

    Retourne (ys, xs, yy, xx, ring_mask) où yy/xx sont les grilles
    complètes et ring_mask le masque de l'anneau.
    """
    H, W = shape
    y_min = max(0, cy - r_outer)
    y_max = min(H, cy + r_outer + 1)
    x_min = max(0, cx - r_outer)
    x_max = min(W, cx + r_outer + 1)
    ys = np.arange(y_min, y_max)
    xs = np.arange(x_min, x_max)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    ring = (d2 > r_inner ** 2) & (d2 <= r_outer ** 2)
    return ys, xs, yy, xx, ring


def sample_ring(ch: np.ndarray, cx: int, cy: int,
                r_inner: int, r_outer: int,
                statistic: str = "mean") -> float:
    """Statistique des pixels dans l'anneau [r_inner, r_outer].

    statistic peut être "mean", "max", "p95", "p90", "std".
    """
    ys, xs, yy, xx, ring = _make_ring_coords(ch.shape, cx, cy, r_inner, r_outer)
    patch = ch[np.ix_(ys, xs)]
    vals = patch[ring]
    if vals.size == 0:
        return 0.0
    if statistic == "mean":
        return float(vals.mean())
    if statistic == "max":
        return float(vals.max())
    if statistic == "p95":
        return float(np.percentile(vals, 95))
    if statistic == "p90":
        return float(np.percentile(vals, 90))
    if statistic == "std":
        return float(vals.std())
    return float(vals.mean())


def sample_disk(ch: np.ndarray, cx: int, cy: int, r: int,
                statistic: str = "mean") -> float:
    """Statistique des pixels dans le disque de rayon r."""
    return sample_ring(ch, cx, cy, 0, r, statistic)


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACTION DE FEATURES POUR ML
# ─────────────────────────────────────────────────────────────────────────────

def radial_profile(
    ch: np.ndarray, cx: int, cy: int, radii: List[int]
) -> List[float]:
    """Profil radial complet en un seul passage.

    Découpe le disque de rayon radii[-1] en bandes concentriques définies
    par radii, et retourne pour chaque bande : mean, std, median, p25, p75.
    """
    max_r = radii[-1]
    ys, xs, yy, xx, mask = _make_ring_coords(ch.shape, cx, cy, 0, max_r)
    patch = ch[np.ix_(ys, xs)]
    vals = patch[mask]
    dists = np.sqrt((yy[mask] - cy) ** 2 + (xx[mask] - cx) ** 2)
    out: List[float] = []
    for r0, r1 in zip(radii[:-1], radii[1:]):
        band = vals[(dists >= r0) & (dists < r1)]
        if len(band) == 0:
            out.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            out.extend([
                float(band.mean()),
                float(band.std()),
                float(np.median(band)),
                float(np.percentile(band, 25)),
                float(np.percentile(band, 75)),
            ])
    return out


def extract_features(
    nuclei: List[NucleusRecord],
    green_u8: np.ndarray,
    red_u8: np.ndarray,
    blue_u8: np.ndarray,
    white_u8: np.ndarray,
    bg_green: np.ndarray,
    bg_red: np.ndarray,
    bg_blue: np.ndarray,
    green_bg_global: float,
    red_bg_global: float,
    blue_bg_global: float,
    cfg: Config,
) -> Tuple[np.ndarray, List[str]]:
    """Extrait un vecteur de features riche pour chaque noyau.

    Retourne (feature_matrix, feature_names).
    """
    radii = sorted(set([0, cfg.nucleus_r // 2, cfg.nucleus_r,
                        cfg.nucleus_r * 2, cfg.nucleus_r * 3, cfg.halo_r]))
    n_bands = len(radii) - 1
    band_stats = ["mean", "std", "median", "p25", "p75"]

    # Construire les noms en premier (même ordre que la boucle row)
    names: List[str] = []
    # métadonnées
    names.extend(["area", "r_est", "bg_loc_G", "bg_loc_R", "bg_loc_B"])
    # profils radiaux bruts + bg-norm pour tous les canaux
    for ch_name in ["G", "R", "B", "W"]:
        for b_idx in range(n_bands):
            for stat in band_stats:
                names.append(f"{ch_name}_b{b_idx}_{stat}")
        for b_idx in range(n_bands):
            for stat in band_stats:
                names.append(f"{ch_name}_b{b_idx}_{stat}_bgnorm")
    # features simples (équivalent seuils)
    names.extend(["G_ring_mean", "R_disk_mean", "R_ring_mean",
                  "B_disk_mean", "B_ring_mean"])
    # ratios inter-canaux
    names.extend(["Gring_Rdisk_ratio", "Gring_Bdisk_ratio",
                  "Rdisk_Bdisk_ratio", "Rring_Bring_ratio"])

    rows: List[List[float]] = []

    for idx, (cx, cy, area, r_est) in enumerate(tqdm(nuclei, desc="Extracting features")):
        loc_g = max(float(bg_green[cy, cx]), green_bg_global * cfg.bg_floor_frac)
        loc_r = max(float(bg_red[cy, cx]),   red_bg_global   * cfg.bg_floor_frac)
        loc_b = max(float(bg_blue[cy, cx]),  blue_bg_global  * cfg.bg_floor_frac)

        row: List[float] = [float(area), float(r_est), loc_g, loc_r, loc_b]

        for ch, bg in [(green_u8, loc_g), (red_u8, loc_r),
                       (blue_u8, loc_b), (white_u8, 0.0)]:
            prof = radial_profile(ch, cx, cy, radii)
            row.extend(prof)
            if bg > 0:
                row.extend([v / bg for v in prof])
            else:
                row.extend([0.0] * len(prof))

        # features simples
        g_ring = radial_profile(green_u8, cx, cy, [cfg.nucleus_r, cfg.halo_r])[0]
        r_disk = radial_profile(red_u8, cx, cy, [0, cfg.nucleus_r])[0]
        r_ring = radial_profile(red_u8, cx, cy, [cfg.nucleus_r, cfg.halo_r])[0]
        b_disk = radial_profile(blue_u8, cx, cy, [0, cfg.nucleus_r])[0]
        b_ring = radial_profile(blue_u8, cx, cy, [cfg.nucleus_r, cfg.halo_r])[0]
        row.extend([g_ring, r_disk, r_ring, b_disk, b_ring])

        # ratios
        eps = 1e-6
        row.append(g_ring / (r_disk + eps))
        row.append(g_ring / (b_disk + eps))
        row.append(r_disk / (b_disk + eps))
        row.append(r_ring / (b_ring + eps))

        rows.append(row)

    return np.array(rows, dtype=np.float32), names


# ─────────────────────────────────────────────────────────────────────────────
#  CLASSIFIEUR ML (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────

class MLClassifier:
    """Classifieur multi-label (G, R, B) basé sur Random Forest.

    Trois modèles binaires indépendants (un par canal).
    """

    def __init__(self, model_path: Optional[str] = None):
        self.models: dict[str, RandomForestClassifier] = {}
        self.feature_names: List[str] = []
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(self, X: np.ndarray, y: np.ndarray,
              feature_names: List[str]) -> dict[str, float]:
        """Entraîne 3 RF binaires (green, red, blue).

        y shape: (n_samples, 3) avec colonnes [green, red, blue].
        Retourne les scores sur le train set.
        """
        self.feature_names = feature_names
        scores = {}
        for idx, label in enumerate(["green", "red", "blue"]):
            m = RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            m.fit(X, y[:, idx])
            self.models[label] = m
            scores[label] = float(m.score(X, y[:, idx]))
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne (n, 3) booléen [green, red, blue]."""
        out = np.zeros((X.shape[0], 3), dtype=bool)
        for idx, label in enumerate(["green", "red", "blue"]):
            if label in self.models:
                out[:, idx] = self.models[label].predict(X).astype(bool)
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne (n, 3) float probabilités [green, red, blue]."""
        out = np.zeros((X.shape[0], 3), dtype=np.float32)
        for idx, label in enumerate(["green", "red", "blue"]):
            if label in self.models:
                out[:, idx] = self.models[label].predict_proba(X)[:, 1]
        return out

    def save(self, path: str) -> None:
        d = {
            "models": self.models,
            "feature_names": self.feature_names,
        }
        with open(path, "wb") as f:
            pickle.dump(d, f)
        print(f"Model saved -> {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.models = d["models"]
        self.feature_names = d["feature_names"]
        print(f"Model loaded <- {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRAÎNEMENT DEPUIS CSV
# ─────────────────────────────────────────────────────────────────────────────

def prepare_training_data(
    csv_path: str,
    validated_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Charge un CSV de features + labels pour l'entraînement.

    Le CSV doit contenir les colonnes de features (préfixe 'f_') et
    les colonnes 'gt_green', 'gt_red', 'gt_blue' (0/1).

    validated_only=True ne conserve que les lignes 'validated'=1, c'est à
    dire les noyaux dont la vérité terrain a été confirmée manuellement
    (via la GUI) plutôt que recopiée depuis la prédiction brute -> permet
    un entraînement supervisé sur une base réellement fiable une fois
    suffisamment de validations accumulées. Sans effet si le CSV ne contient
    pas de colonne 'validated' (rétrocompat avec les anciens exports).
    """
    rows: List[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError(f"Aucune ligne dans {csv_path}")

    if validated_only and "validated" in rows[0]:
        before = len(rows)
        rows = [r for r in rows if str(r.get("validated", "0")) == "1"]
        print(f"Filtrage 'validated'=1 : {len(rows)}/{before} lignes conservées.")
        if not rows:
            raise ValueError(
                "Aucune ligne 'validated'=1 dans ce CSV. Validez des noyaux "
                "dans la GUI (clic sur un cercle) puis ré-exportez, ou "
                "relancez sans validated_only."
            )

    feature_cols = [k for k in rows[0] if k.startswith("f_")]
    if not feature_cols:
        raise ValueError(
            "Aucune colonne 'f_' trouvée. "
            "Le CSV doit contenir les features préfixées par 'f_'."
        )

    X = np.array([[float(r[c]) for c in feature_cols] for r in rows], dtype=np.float32)
    y = np.zeros((len(rows), 3), dtype=np.int32)
    for idx, label in enumerate(["gt_green", "gt_red", "gt_blue"]):
        if label in rows[0]:
            y[:, idx] = np.array([int(r[label]) for r in rows])

    n_labels = y.sum(axis=0)
    print(f"Labels chargés : green={n_labels[0]}, red={n_labels[1]}, blue={n_labels[2]} "
          f"sur {len(rows)} cellules")
    return X, y, feature_cols


def export_features_csv(
    csv_path: str,
    features: np.ndarray,
    feature_names: List[str],
    results: List[ClassificationResult],
    nuclei: List[NucleusRecord],
    corrections: Optional[Dict[int, "Correction"]] = None,
) -> None:
    """Exporte les features + prédictions dans un CSV éditable pour
    entraînement supervisé.

    `results` doit être la PRÉDICTION BRUTE (threshold ou ML), pas une
    version déjà corrigée -> colonnes pred_*.

    Si `corrections` est fourni (vérité terrain validée manuellement, voir
    Correction / la GUI "cliquer sur un noyau"), les colonnes gt_* sont
    remplies avec cette vérité terrain réelle et 'validated'=1. Pour les
    noyaux non validés manuellement, gt_* reprend la prédiction brute par
    défaut (comportement historique, repli) mais 'validated'=0 : à filtrer
    ou pondérer différemment lors d'un entraînement supervisé fiable — seule
    la colonne 'validated'=1 constitue une vérité terrain confirmée par un
    humain.
    """
    corrections = corrections or {}
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["cx", "cy", "area", "r_est"] + \
                 [f"f_{n}" for n in feature_names] + \
                 ["pred_green", "pred_red", "pred_blue",
                  "gt_green", "gt_red", "gt_blue", "validated"]
        writer.writerow(header)
        for i, (cx, cy, area, r_est) in enumerate(nuclei):
            is_g, is_r, is_b = results[i][2], results[i][3], results[i][4]
            corr = corrections.get(i)
            if corr is not None:
                gt_g, gt_r, gt_b = corr.gt_green, corr.gt_red, corr.gt_blue
                validated = 1
            else:
                gt_g, gt_r, gt_b = is_g, is_r, is_b
                validated = 0
            row = [cx, cy, area, r_est] + \
                  [f"{v:.4f}" for v in features[i].tolist()] + \
                  [int(is_g), int(is_r), int(is_b),
                   int(gt_g), int(gt_r), int(gt_b), validated]
            writer.writerow(row)
    n_validated = sum(1 for i in range(len(nuclei)) if i in corrections)
    print(f"Features exportees -> {csv_path}")
    print(f"  {n_validated} noyaux valides manuellement / {len(nuclei)} "
          f"(colonne 'validated'=1 -> verite terrain fiable pour l'entrainement)")
    if n_validated < len(nuclei):
        print(f"  (Les {len(nuclei) - n_validated} lignes 'validated'=0 recopient la "
              f"prediction brute ; corrigez-les manuellement si besoin puis --train {csv_path})")


# ─────────────────────────────────────────────────────────────────────────────
#  DÉTECTION DES NOYAUX
# ─────────────────────────────────────────────────────────────────────────────

NucleusRecord = Tuple[int, int, int, int]  # cx, cy, area, r_est


def detect_nuclei(white_u8: np.ndarray, cfg: Config) -> Tuple[List[NucleusRecord], int, List, float]:
    """Détecte les noyaux par seuillage Otsu + watershed si nécessaire.

    Retourne :
        nuclei       – liste de (cx, cy, area, r_est)
        n_ws_split   – nombre de blobs découpés par watershed
        blobs        – blobs initiaux (pour stats)
        median_single– aire médiane d'un noyau isolé
    """
    blur = cv2.GaussianBlur(white_u8, (5, 5), 1)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(th, cv2.MORPH_OPEN, k3, iterations=1)

    num_labels, label_img, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    blobs: List = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if cfg.min_area <= area <= cfg.max_area:
            blobs.append((i, area, int(centroids[i][0]), int(centroids[i][1])))

    areas = np.array([b[1] for b in blobs])
    cutoff = np.percentile(areas, 70)
    median_single = float(np.median(areas[areas <= cutoff]))

    nuclei: List[NucleusRecord] = []
    n_ws_split = 0

    for lid, area, cx, cy in blobs:
        n_expected = (
            1 if area < cfg.merge_ratio * median_single
            else max(1, round(area / median_single))
        )
        if n_expected == 1:
            r_est = max(4, int(np.sqrt(area / np.pi)))
            nuclei.append((cx, cy, area, r_est))
        else:
            bx = int(stats[lid, cv2.CC_STAT_LEFT])
            by = int(stats[lid, cv2.CC_STAT_TOP])
            bw = int(stats[lid, cv2.CC_STAT_WIDTH])
            bh = int(stats[lid, cv2.CC_STAT_HEIGHT])
            blob_mask = (label_img[by:by + bh, bx:bx + bw] == lid).astype(np.uint8)
            sub = _split_blob_watershed(blob_mask, bx, by, n_expected, cfg)
            if sub:
                n_ws_split += 1
                for scx, scy, sarea in sub:
                    r_est = max(4, int(np.sqrt(sarea / np.pi)))
                    nuclei.append((scx, scy, sarea, r_est))
            else:
                r_est = max(4, int(np.sqrt(area / np.pi)))
                for _ in range(n_expected):
                    nuclei.append((cx, cy, area // n_expected, r_est))

    return nuclei, n_ws_split, blobs, median_single


def _split_blob_watershed(
    blob_mask_full: np.ndarray,
    blob_x: int, blob_y: int,
    expected_n: int,
    cfg: Config,
) -> List[Tuple[int, int, int]]:
    """Découpe un blob fusionné par watershed.

    Retourne une liste de (cx, cy, area) en coordonnées image globale.
    """
    dist = cv2.distanceTransform(blob_mask_full, cv2.DIST_L2, 5)
    dist_s = cv2.GaussianBlur(dist, (5, 5), 1)
    H_b, W_b = blob_mask_full.shape

    flat_idx = np.argsort(dist_s.ravel())[::-1]
    suppress = np.zeros(H_b * W_b, dtype=bool)
    peaks: List[Tuple[int, int]] = []
    min_sep2 = (cfg.nucleus_r * 1.5) ** 2
    sup_radius = cfg.nucleus_r * 2
    for fi in flat_idx:
        if suppress[fi]:
            continue
        if dist_s.ravel()[fi] < 0.1 * dist_s.max():
            break
        py, px = divmod(fi, W_b)
        y_min = max(0, py - sup_radius)
        y_max = min(H_b, py + sup_radius + 1)
        x_min = max(0, px - sup_radius)
        x_max = min(W_b, px + sup_radius + 1)
        ys_ = np.arange(y_min, y_max)
        xs_ = np.arange(x_min, x_max)
        yy_, xx_ = np.meshgrid(ys_, xs_, indexing="ij")
        nbr = (yy_ - py) ** 2 + (xx_ - px) ** 2 < min_sep2
        idx_nbr = (yy_[nbr] * W_b + xx_[nbr]).ravel()
        suppress[idx_nbr] = True
        peaks.append((py, px))
        if len(peaks) >= expected_n * 2:
            break

    if len(peaks) < 2:
        M = cv2.moments(blob_mask_full)
        if M["m00"] == 0:
            return []
        cxc = int(M["m10"] / M["m00"]) + blob_x
        cyc = int(M["m01"] / M["m00"]) + blob_y
        return [(cxc, cyc, int(blob_mask_full.sum()))]

    markers = np.zeros((H_b, W_b), dtype=np.int32)
    for k, (py, px) in enumerate(peaks, 1):
        markers[py, px] = k

    ws_in = cv2.cvtColor(
        cv2.normalize(dist_s, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLOR_GRAY2BGR,
    )
    ws_labels = cv2.watershed(ws_in, markers)

    result: List[Tuple[int, int, int]] = []
    for k in range(1, len(peaks) + 1):
        seg = (ws_labels == k).astype(np.uint8) & blob_mask_full
        area_k = int(seg.sum())
        if area_k < cfg.min_area // 2:
            continue
        M = cv2.moments(seg)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"]) + blob_x
        cy = int(M["m01"] / M["m00"]) + blob_y
        result.append((cx, cy, area_k))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

ClassificationResult = Tuple[int, int, bool, bool, bool, int]  # cx, cy, is_green, is_red, is_blue, r_est


# ─────────────────────────────────────────────────────────────────────────────
#  CORRECTIONS MANUELLES (validation humaine, vérité terrain)
# ─────────────────────────────────────────────────────────────────────────────
#
# Objectif : permettre à l'utilisateur de cliquer sur un noyau dans l'aperçu,
# de confirmer ou d'infirmer la prédiction, et de faire propager cette
# correction jusqu'au rendu / rapport / export CSV. Contrairement à
# l'auto-entraînement existant (qui recopiait aveuglément la prédiction
# threshold comme "vérité terrain"), ceci construit une base de vérité
# terrain réellement validée par un humain, exploitable plus tard pour un
# entraînement supervisé fiable du Random Forest (ou tout autre modèle).
#
# Un noyau est identifié par son index dans la liste `nuclei`/`results`
# (stable tant que la détection en cache n'est pas relancée).

@dataclass
class Correction:
    """Vérité terrain confirmée manuellement pour un noyau donné."""
    gt_green: bool
    gt_red: bool
    gt_blue: bool
    was_prediction_correct: bool   # True si l'utilisateur a validé la prédiction telle quelle
    corrected: bool                # True si la classe a été modifiée manuellement (cas "Faux")
    validated_at: str              # horodatage ISO 8601

    def label(self) -> str:
        return classification_label(self.gt_green, self.gt_red, self.gt_blue)


def classification_label(is_green: bool, is_red: bool, is_blue: bool) -> str:
    """Libellé lisible d'une combinaison de classes, pour l'affichage GUI."""
    parts = [n for n, v in (("GREEN", is_green), ("RED", is_red), ("BLUE", is_blue)) if v]
    return " + ".join(parts) if parts else "Non classé"


def apply_corrections(
    results: List[ClassificationResult],
    corrections: Optional[Dict[int, Correction]],
) -> List[ClassificationResult]:
    """Remplace, pour les noyaux validés manuellement, la prédiction brute par
    la vérité terrain confirmée par l'utilisateur. Les coordonnées (cx, cy)
    et r_est sont conservés ; seuls les labels g/r/b changent.

    Ne modifie jamais `results` en place (retourne une nouvelle liste), afin
    que la liste de prédictions brutes reste disponible pour l'export CSV
    (colonnes pred_*) même après application des corrections pour l'affichage.
    """
    if not corrections:
        return results
    out: List[ClassificationResult] = []
    for idx, (cx, cy, g, r, b, r_est) in enumerate(results):
        c = corrections.get(idx)
        if c is None:
            out.append((cx, cy, g, r, b, r_est))
        else:
            out.append((cx, cy, c.gt_green, c.gt_red, c.gt_blue, r_est))
    return out


def rerender_with_corrections(
    composite: np.ndarray,
    base_results: List[ClassificationResult],
    corrections: Optional[Dict[int, Correction]],
    cfg: Config,
) -> dict:
    """Applique les corrections manuelles à des résultats déjà classés puis
    reconstruit uniquement le rendu/comptage (aucun recalcul de seuils, aucun
    rééchantillonnage radial). Étape TRÈS RAPIDE, pensée pour donner un retour
    visuel instantané juste après une validation manuelle dans la GUI, sans
    relancer classify_and_render() (qui repasse par tous les noyaux)."""
    corrected = apply_corrections(base_results, corrections)
    visuals = render_annotated_visuals(composite, corrected, cfg)
    counts = summarize_counts(corrected)
    return {
        "results": corrected,
        "visuals": visuals,
        "counts": counts,
        "total": len(corrected),
    }


def find_nearest_nucleus(
    results: List[ClassificationResult],
    x: float, y: float,
    max_dist: Optional[float] = None,
) -> Optional[int]:
    """Retourne l'index (dans `results`/`nuclei`, même ordre) du noyau le plus
    pertinent pour un clic en (x, y) [coordonnées image pleine résolution],
    ou None si aucun noyau n'est à proximité raisonnable.

    Gère les noyaux imbriqués/qui se chevauchent : si le point cliqué tombe
    réellement DANS le disque d'un ou plusieurs noyaux, on retient le PLUS
    PETIT d'entre eux (le plus "spécifique", généralement rendu au-dessus et
    le plus probable candidat d'un clic précis) plutôt que simplement le
    centre le plus proche — sinon cliquer sur un petit noyau contenu dans un
    plus grand sélectionnerait toujours le grand. Si aucun disque ne
    contient le point, on retombe sur le centre le plus proche, avec une
    tolérance minimale de clic (max(max_dist, r_est + 6) par noyau).
    """
    inside_candidates: List[Tuple[int, int]] = []  # (r_est, idx), triés ensuite par rayon croissant
    best_idx: Optional[int] = None
    best_d2: Optional[float] = None
    for idx, (cx, cy, _g, _r, _b, r_est) in enumerate(results):
        d2 = (cx - x) ** 2 + (cy - y) ** 2
        if d2 <= r_est ** 2:
            inside_candidates.append((r_est, idx))
        limit = max(max_dist, r_est + 6) if max_dist is not None else (r_est + 6)
        if d2 <= limit ** 2 and (best_d2 is None or d2 < best_d2):
            best_d2 = d2
            best_idx = idx
    if inside_candidates:
        inside_candidates.sort(key=lambda t: t[0])
        return inside_candidates[0][1]
    return best_idx


def save_corrections(path: str, corrections: Dict[int, Correction], meta: Optional[dict] = None) -> None:
    """Sauvegarde la base de vérité terrain validée manuellement (JSON)."""
    data = {
        "meta": meta or {},
        "corrections": {
            str(idx): {
                "gt_green": c.gt_green, "gt_red": c.gt_red, "gt_blue": c.gt_blue,
                "was_prediction_correct": c.was_prediction_correct,
                "corrected": c.corrected,
                "validated_at": c.validated_at,
            }
            for idx, c in corrections.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_corrections(path: str) -> Tuple[Dict[int, Correction], dict]:
    """Charge une base de vérité terrain sauvegardée. Retourne (corrections, meta)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("corrections", data)  # tolère aussi un fichier "à plat" (rétrocompat)
    meta = data.get("meta", {})
    corrections: Dict[int, Correction] = {}
    for idx_str, d in raw.items():
        corrections[int(idx_str)] = Correction(
            gt_green=bool(d["gt_green"]), gt_red=bool(d["gt_red"]), gt_blue=bool(d["gt_blue"]),
            was_prediction_correct=bool(d.get("was_prediction_correct", True)),
            corrected=bool(d.get("corrected", False)),
            validated_at=d.get("validated_at", ""),
        )
    return corrections, meta


def corrections_summary(corrections: Dict[int, Correction]) -> dict:
    """Petit résumé pour affichage GUI : total validé / dont corrigés."""
    total = len(corrections)
    corrected = sum(1 for c in corrections.values() if c.corrected)
    confirmed = total - corrected
    return {"total": total, "confirmed": confirmed, "corrected": corrected}


def classify_nuclei(
    nuclei: List[NucleusRecord],
    green_u8: np.ndarray,
    red_u8: np.ndarray,
    blue_u8: np.ndarray,
    bg_green: np.ndarray,
    bg_red: np.ndarray,
    bg_blue: np.ndarray,
    green_bg_global: float,
    red_bg_global: float,
    blue_bg_global: float,
    cfg: Config,
    ml: Optional[MLClassifier] = None,
    features: Optional[np.ndarray] = None,
) -> List[ClassificationResult]:
    """Classe chaque noyau par échantillonnage radial (threshold) ou ML."""
    results: List[ClassificationResult] = []

    if ml is not None and features is not None:
        preds = ml.predict(features)
        for i, (cx, cy, area, r_est) in enumerate(nuclei):
            results.append((cx, cy, bool(preds[i, 0]), bool(preds[i, 1]),
                            bool(preds[i, 2]), r_est))
        return results

    for idx, (cx, cy, area, r_est) in enumerate(tqdm(nuclei, desc="Classification")):
        loc_g = max(float(bg_green[cy, cx]), green_bg_global * cfg.bg_floor_frac)
        loc_r = max(float(bg_red[cy, cx]),   red_bg_global   * cfg.bg_floor_frac)
        loc_b = max(float(bg_blue[cy, cx]),  blue_bg_global  * cfg.bg_floor_frac)

        g_ring = sample_ring(green_u8, cx, cy, cfg.nucleus_r, cfg.halo_r)
        is_green = (g_ring >= cfg.green_thresh) and (g_ring >= cfg.green_bg_ratio * loc_g)

        r_inner = sample_disk(red_u8, cx, cy, cfg.nucleus_r, cfg.red_stat)
        r_outer = sample_ring(red_u8, cx, cy, cfg.nucleus_r, cfg.halo_r, cfg.red_stat)
        r_best = max(r_inner, r_outer)
        is_red = (r_best >= cfg.red_thresh) and (r_best >= cfg.red_bg_ratio * loc_r)
        if cfg.red_not_greener and is_red:
            g_disk = sample_disk(green_u8, cx, cy, cfg.halo_r)
            is_red = r_best > g_disk

        b_inner = sample_disk(blue_u8, cx, cy, cfg.nucleus_r)
        b_outer = sample_ring(blue_u8, cx, cy, cfg.nucleus_r, cfg.halo_r)
        b_best = max(b_inner, b_outer)
        is_blue = (b_best >= cfg.blue_thresh) and (b_best >= cfg.blue_bg_ratio * loc_b)

        results.append((cx, cy, is_green, is_red, is_blue, r_est))
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def build_composite(
    green_u8: np.ndarray,
    red_u8: np.ndarray,
    blue_u8: np.ndarray,
    white_u8: np.ndarray,
) -> np.ndarray:
    """Assemble les 4 canaux en une image de fond fausses couleurs (BGR)."""
    H, W = white_u8.shape
    composite = np.zeros((H, W, 3), dtype=np.uint8)
    composite[:, :, 1] = (green_u8 * 0.6).astype(np.uint8)
    composite[:, :, 2] = (red_u8 * 0.6).astype(np.uint8)
    composite[:, :, 0] = (blue_u8 * 0.6).astype(np.uint8)
    w_contrib = (white_u8 * 0.4).astype(np.uint8)
    composite = np.clip(
        composite.astype(np.int16) + w_contrib[:, :, None], 0, 255
    ).astype(np.uint8)
    return composite


def draw_fixed(canvas: np.ndarray, cells_xy: List[Tuple[int, int]],
               colour: Tuple[int, int, int], r: int = 8, thickness: int = 2) -> None:
    """Dessine des cercles de rayon fixe."""
    for cx, cy in cells_xy:
        cv2.circle(canvas, (cx, cy), r, colour, thickness)


def draw_prop(canvas: np.ndarray, cells_r: List[Tuple[int, int, int]],
              colour: Tuple[int, int, int], thickness: int = 2) -> None:
    """Dessine des cercles de rayon proportionnel à r_est.

    Trie du plus grand au plus petit rayon avant de dessiner : les noyaux
    imbriqués/qui se chevauchent (un petit noyau contenu dans le halo d'un
    plus grand) restent ainsi visibles par-dessus, au lieu d'être recouverts
    par le contour d'un noyau voisin plus grand dessiné après eux.
    """
    for cx, cy, r_est in sorted(cells_r, key=lambda t: -t[2]):
        cv2.circle(canvas, (cx, cy), max(6, r_est + 3), colour, thickness)


def add_legend(canvas: np.ndarray, entries: List[Tuple[str, Tuple[int, int, int], int]],
               x0: int = 12, y0: int = 20, lh: int = 22, sq: int = 14) -> None:
    """Ajoute une légende avec carrés colorés et comptes."""
    for label, color, count in entries:
        cv2.rectangle(canvas, (x0, y0 - sq + 2), (x0 + sq, y0 + 2), color, -1)
        cv2.putText(canvas, f"{label}  ({count})",
                    (x0 + sq + 6, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
        y0 += lh


def render_annotated_visuals(
    composite: np.ndarray,
    results: List[ClassificationResult],
    cfg: Config,
) -> dict:
    """Construit les images annotées EN MÉMOIRE (aucune écriture disque).

    Retourne {nom_fichier_logique: image np.ndarray BGR}. Utilisé aussi bien
    pour l'écriture finale (save_annotated_images) que pour un aperçu live
    dans une interface graphique (réglage interactif des seuils).
    """
    green_cells  = [(cx, cy) for cx, cy, g, r, b, _ in results if g]
    red_cells    = [(cx, cy) for cx, cy, g, r, b, _ in results if r]
    blue_cells   = [(cx, cy) for cx, cy, g, r, b, _ in results if b]
    unclassified = [(cx, cy) for cx, cy, g, r, b, _ in results if not g and not r and not b]

    only_green = [(cx, cy, re) for cx, cy, g, r, b, re in results if g and not r and not b]
    only_red   = [(cx, cy, re) for cx, cy, g, r, b, re in results if not g and r and not b]
    only_blue  = [(cx, cy, re) for cx, cy, g, r, b, re in results if not g and not r and b]
    green_red  = [(cx, cy, re) for cx, cy, g, r, b, re in results if g and r and not b]
    green_blue = [(cx, cy, re) for cx, cy, g, r, b, re in results if g and not r and b]
    red_blue   = [(cx, cy, re) for cx, cy, g, r, b, re in results if not g and r and b]
    all_three  = [(cx, cy, re) for cx, cy, g, r, b, re in results if g and r and b]

    visuals: dict = {}

    for fname, cells_xy, col, label in [
        ("annotated_green.png", green_cells, cfg.c_green, "GREEN (AQP2)"),
        ("annotated_red.png",   red_cells,   cfg.c_red,   "RED (AE1)"),
        ("annotated_blue.png",  blue_cells,  cfg.c_blue,  "BLUE"),
    ]:
        vis = composite.copy()
        draw_fixed(vis, cells_xy, col, r=8)
        add_legend(vis, [(label, col, len(cells_xy))])
        visuals[fname] = vis

    vis_all = composite.copy()
    draw_prop(vis_all, only_green, cfg.c_green)
    draw_prop(vis_all, only_red,   cfg.c_red)
    draw_prop(vis_all, only_blue,  cfg.c_blue)
    draw_fixed(vis_all, unclassified, cfg.c_unc, r=4, thickness=1)
    draw_prop(vis_all, green_red,  cfg.c_gr)
    draw_prop(vis_all, green_blue, cfg.c_gb)
    draw_prop(vis_all, red_blue,   cfg.c_rb)
    draw_prop(vis_all, all_three,  cfg.c_grb)

    add_legend(vis_all, [
        ("GREEN seul (AQP2)", cfg.c_green, len(only_green)),
        ("RED seul (AE1)",    cfg.c_red,   len(only_red)),
        ("BLUE seul",         cfg.c_blue,  len(only_blue)),
        ("GREEN + RED",       cfg.c_gr,    len(green_red)),
        ("GREEN + BLUE",      cfg.c_gb,    len(green_blue)),
        ("RED + BLUE",        cfg.c_rb,    len(red_blue)),
        ("G + R + B",         cfg.c_grb,   len(all_three)),
        ("Non classifie",     cfg.c_unc,   len(unclassified)),
    ])
    visuals["annotated_all.png"] = vis_all
    return visuals


def summarize_counts(results: List[ClassificationResult]) -> dict:
    """Calcule les comptages de co-expression à partir des résultats de classification.

    Utilisé à la fois par generate_report() et par l'aperçu live de la GUI.
    """
    total = len(results)
    green_cells  = sum(1 for _, _, g, r, b, _ in results if g)
    red_cells    = sum(1 for _, _, g, r, b, _ in results if r)
    blue_cells   = sum(1 for _, _, g, r, b, _ in results if b)
    unclassified = sum(1 for _, _, g, r, b, _ in results if not g and not r and not b)
    only_green  = sum(1 for _, _, g, r, b, _ in results if g and not r and not b)
    only_red    = sum(1 for _, _, g, r, b, _ in results if not g and r and not b)
    only_blue   = sum(1 for _, _, g, r, b, _ in results if not g and not r and b)
    green_red   = sum(1 for _, _, g, r, b, _ in results if g and r and not b)
    green_blue  = sum(1 for _, _, g, r, b, _ in results if g and not r and b)
    red_blue    = sum(1 for _, _, g, r, b, _ in results if not g and r and b)
    all_three   = sum(1 for _, _, g, r, b, _ in results if g and r and b)
    return {
        "total": total, "green": green_cells, "red": red_cells, "blue": blue_cells,
        "unclassified": unclassified, "only_green": only_green, "only_red": only_red,
        "only_blue": only_blue, "green_red": green_red, "green_blue": green_blue,
        "red_blue": red_blue, "all_three": all_three,
    }


def save_annotated_images(
    composite: np.ndarray,
    results: List[ClassificationResult],
    cfg: Config,
    output_dir: Path,
) -> dict:
    """Rend puis écrit sur disque les images annotées (individuelles + combinée).

    Retourne un dict {nom_logique: chemin_absolu} pour affichage GUI.
    """
    visuals = render_annotated_visuals(composite, results, cfg)
    paths: dict = {}
    for fname, vis in visuals.items():
        out_path = output_dir / fname
        cv2.imwrite(str(out_path), vis)
        paths[fname] = str(out_path)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
#  RAPPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    cfg: Config,
    W: int, H: int,
    blobs: List,
    median_single: float,
    n_ws_split: int,
    total: int,
    results: List[ClassificationResult],
    correction_stats: Optional[dict] = None,
) -> str:
    """Génère le rapport texte."""
    def pct(n: int) -> str:
        return f"{100 * n / total:.1f}%" if total > 0 else "N/A"

    green_cells  = sum(1 for _, _, g, r, b, _ in results if g)
    red_cells    = sum(1 for _, _, g, r, b, _ in results if r)
    blue_cells   = sum(1 for _, _, g, r, b, _ in results if b)
    unclassified = sum(1 for _, _, g, r, b, _ in results if not g and not r and not b)

    only_green  = sum(1 for _, _, g, r, b, _ in results if g and not r and not b)
    only_red    = sum(1 for _, _, g, r, b, _ in results if not g and r and not b)
    only_blue   = sum(1 for _, _, g, r, b, _ in results if not g and not r and b)
    green_red   = sum(1 for _, _, g, r, b, _ in results if g and r and not b)
    green_blue  = sum(1 for _, _, g, r, b, _ in results if g and not r and b)
    red_blue    = sum(1 for _, _, g, r, b, _ in results if not g and r and b)
    all_three   = sum(1 for _, _, g, r, b, _ in results if g and r and b)

    validation_section = ""
    if correction_stats is not None:
        validation_section = f"""
-- VALIDATION MANUELLE --
  Noyaux valides manuellement : {correction_stats['total']:6d}  ({pct(correction_stats['total'])})
    dont predictions confirmees : {correction_stats['confirmed']:6d}
    dont predictions corrigees  : {correction_stats['corrected']:6d}
  Les comptages ci-dessus integrent ces corrections (verite terrain > prediction brute).
"""

    return f"""
==============================================
     CELL CLASSIFICATION REPORT  v5
==============================================
Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Source    : {cfg.czi_path}
Resolution: {W}x{H} px

-- DETECTION --
  Blobs                   : {len(blobs)}
  Aire mediane noyau seul : {median_single:.0f} px2
  Blobs coupes watershed  : {n_ws_split}
  TOTAL noyaux            : {total}

-- CLASSIFICATION MULTI-LABEL --
  GREEN  (AQP2) : {green_cells:6d}  ({pct(green_cells)})
  RED    (AE1)  : {red_cells:6d}  ({pct(red_cells)})
  BLUE          : {blue_cells:6d}  ({pct(blue_cells)})
  Non classes   : {unclassified:6d}  ({pct(unclassified)})

-- CO-EXPRESSION --
  GREEN seul    : {only_green:6d}  ({pct(only_green)})
  RED   seul    : {only_red:6d}  ({pct(only_red)})
  BLUE  seul    : {only_blue:6d}  ({pct(only_blue)})
  GREEN+RED     : {green_red:6d}  ({pct(green_red)})
  GREEN+BLUE    : {green_blue:6d}  ({pct(green_blue)})
  RED+BLUE      : {red_blue:6d}  ({pct(red_blue)})
  G+R+B         : {all_three:6d}  ({pct(all_three)})
  TOTAL         : {total:6d}
{validation_section}
-- PARAMETRES v5 --
  Canaux  blanc={cfg.ch_white} vert={cfg.ch_green} rouge={cfg.ch_red} bleu={cfg.ch_blue}
  NUCLEUS_R={cfg.nucleus_r}px  HALO_R={cfg.halo_r}px

  GREEN  anneau[{cfg.nucleus_r},{cfg.halo_r}]  seuil>={cfg.green_thresh}  ratio>={cfg.green_bg_ratio}x fond
  RED    {cfg.red_stat}(disque[0,{cfg.nucleus_r}], anneau[{cfg.nucleus_r},{cfg.halo_r}])  seuil>={cfg.red_thresh}  ratio>={cfg.red_bg_ratio}x fond
  BLUE   max(disque[0,{cfg.nucleus_r}], anneau[{cfg.nucleus_r},{cfg.halo_r}])  seuil>={cfg.blue_thresh}  ratio>={cfg.blue_bg_ratio}x fond
  Fond plancher={cfg.bg_floor_frac}x mediane globale
  BG_GRID={cfg.bg_grid}x{cfg.bg_grid}  MAX_DIM={cfg.max_dim}
  NORM_CLIP=[{cfg.norm_clip_low}, {cfg.norm_clip_high}] percentile (robuste, remplace min/max brut)

-- FICHIERS --
  annotated_green.png  annotated_red.png  annotated_blue.png
  annotated_all.png    cell_report_v4.txt
"""


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE DÉCOUPLÉ (chargement/détection lents vs classification rapide)
# ─────────────────────────────────────────────────────────────────────────────

# Paramètres qui nécessitent de RECHARGER le CZI et de RE-DÉTECTER les noyaux.
# Tous les autres paramètres (seuils, ratios, bg_grid, bg_floor_frac...) ne
# concernent que la classification et peuvent être recalculés rapidement à
# partir des données déjà en mémoire (cf. classify_and_render).
SLOW_PARAM_NAMES: Tuple[str, ...] = (
    "czi_path", "ch_white", "ch_green", "ch_red", "ch_blue", "max_dim",
    "norm_clip_low", "norm_clip_high",
    "min_area", "max_area", "merge_ratio", "nucleus_r", "halo_r",
)


def slow_params_snapshot(cfg: Config) -> dict:
    """Extrait les paramètres 'lents' de cfg, pour détecter s'il faut redétecter."""
    return {name: getattr(cfg, name) for name in SLOW_PARAM_NAMES}


def slow_params_changed(cfg: Config, cached: dict) -> bool:
    """True si un paramètre nécessitant un rechargement/redétection a changé."""
    return slow_params_snapshot(cfg) != cached.get("slow_params")


def load_and_detect(cfg: Config) -> dict:
    """Charge le CZI et détecte les noyaux (étape LENTE, indépendante des seuils
    de classification G/R/B).

    Retourne un dict réutilisable par classify_and_render() et run(), et
    servant de cache pour un mode de réglage interactif dans une GUI.
    """
    if not Path(cfg.czi_path).exists():
        raise FileNotFoundError(f"Fichier introuvable : {cfg.czi_path}")
    try:
        img = np.squeeze(imread(cfg.czi_path))
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du CZI : {e}") from e
    print(f"Shape brut : {img.shape}  dtype: {img.dtype}")

    if img.ndim < 3:
        raise ValueError("L'image chargée n'a pas assez de dimensions (attendu >= 3)")

    white_u8 = to_uint8(img[cfg.ch_white], cfg.max_dim, cfg.norm_clip_low, cfg.norm_clip_high)
    green_u8 = to_uint8(img[cfg.ch_green], cfg.max_dim, cfg.norm_clip_low, cfg.norm_clip_high)
    red_u8   = to_uint8(img[cfg.ch_red],   cfg.max_dim, cfg.norm_clip_low, cfg.norm_clip_high)
    blue_u8  = to_uint8(img[cfg.ch_blue],  cfg.max_dim, cfg.norm_clip_low, cfg.norm_clip_high)
    H, W = white_u8.shape
    print(f"Resolution retenue : {W}x{H} px")

    composite = build_composite(green_u8, red_u8, blue_u8, white_u8)

    nuclei, n_ws_split, blobs, median_single = detect_nuclei(white_u8, cfg)
    print(f"Noyaux : {len(nuclei)}  (dont {n_ws_split} blobs decoupes par watershed)")

    return {
        "W": W, "H": H,
        "white_u8": white_u8, "green_u8": green_u8, "red_u8": red_u8, "blue_u8": blue_u8,
        "composite": composite,
        "nuclei": nuclei, "n_ws_split": n_ws_split, "blobs": blobs,
        "median_single": median_single,
        "slow_params": slow_params_snapshot(cfg),
    }


def classify_and_render(
    cfg: Config,
    cached: dict,
    corrections: Optional[Dict[int, Correction]] = None,
) -> dict:
    """Reclassifie les noyaux déjà détectés avec les seuils courants et
    construit les images annotées EN MÉMOIRE (aucune écriture disque).

    Étape RAPIDE : ne recharge pas le CZI, ne redétecte pas les noyaux.
    Idéal pour un réglage interactif des seuils dans une GUI.

    Si `corrections` est fourni (validations manuelles, voir Correction),
    l'affichage/comptage ("results", "visuals", "counts") reflète la vérité
    terrain validée plutôt que la prédiction brute, mais la prédiction brute
    reste disponible sous "raw_results" (utile pour l'export CSV : colonnes
    pred_* vs gt_*).
    """
    green_u8 = cached["green_u8"]
    red_u8   = cached["red_u8"]
    blue_u8  = cached["blue_u8"]
    nuclei   = cached["nuclei"]
    composite = cached["composite"]

    bg_green = build_bg_map(green_u8, cfg.bg_grid)
    bg_red   = build_bg_map(red_u8,   cfg.bg_grid)
    bg_blue  = build_bg_map(blue_u8,  cfg.bg_grid)

    green_bg_global = float(np.median(green_u8))
    red_bg_global   = float(np.median(red_u8))
    blue_bg_global  = float(np.median(blue_u8))

    raw_results = classify_nuclei(
        nuclei, green_u8, red_u8, blue_u8,
        bg_green, bg_red, bg_blue,
        green_bg_global, red_bg_global, blue_bg_global,
        cfg,
    )
    display_results = apply_corrections(raw_results, corrections)

    visuals = render_annotated_visuals(composite, display_results, cfg)
    counts = summarize_counts(display_results)

    return {
        "results": display_results, "raw_results": raw_results,
        "visuals": visuals, "counts": counts,
        "total": len(nuclei),
        "bg_maps": (bg_green, bg_red, bg_blue),
        "bg_globals": (green_bg_global, red_bg_global, blue_bg_global),
    }




# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: Config, parsed: ParsedArgs, output_dir: Optional[str] = None,
        cached: Optional[dict] = None,
        corrections: Optional[Dict[int, Correction]] = None) -> dict:
    """Exécute le pipeline complet : chargement → détection → classification → sorties.

    Args:
        cfg: configuration des paramètres d'analyse.
        parsed: namespace avec au minimum `self_train` (bool) et `model_path` (str).
        output_dir: dossier où écrire les images annotées et le rapport.
            Si None, utilise le dossier courant (comportement CLI d'origine).
        cached: résultat optionnel d'un load_and_detect() précédent (obtenu par
            exemple via un mode de réglage interactif dans une GUI). S'il est
            fourni ET que les paramètres 'lents' (voir SLOW_PARAM_NAMES) n'ont
            pas changé depuis, le chargement du CZI et la détection des noyaux
            sont réutilisés tels quels, ce qui accélère beaucoup l'exécution.
        corrections: vérité terrain validée manuellement (voir Correction),
            typiquement construite dans la GUI par clic sur les noyaux en mode
            réglage interactif. Si fourni, les images annotées et le rapport
            reflètent la vérité terrain validée plutôt que la prédiction brute
            pour les noyaux concernés ; la prédiction brute reste disponible
            dans le résultat retourné sous "raw_results" (utile pour l'export
            CSV features + gt, colonnes pred_* vs gt_*).

    Retourne un dict résumé exploitable par une interface (GUI/CLI) :
        {"total": int, "results": [...], "raw_results": [...],
         "image_paths": {...}, "report": str, "report_path": str}

    Lève une exception (au lieu de sys.exit) en cas d'erreur, pour laisser
    l'appelant (CLI ou GUI) décider de la présentation de l'erreur.
    """
    out_dir = Path(output_dir) if output_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    if cached is not None and not slow_params_changed(cfg, cached):
        print("Réutilisation du chargement/détection déjà effectués "
              "(paramètres de détection inchangés) ...")
        data = cached
    else:
        print(f"Loading CZI from {cfg.czi_path} ...")
        data = load_and_detect(cfg)

    white_u8 = data["white_u8"]
    green_u8 = data["green_u8"]
    red_u8   = data["red_u8"]
    blue_u8  = data["blue_u8"]
    composite = data["composite"]
    nuclei = data["nuclei"]
    n_ws_split = data["n_ws_split"]
    blobs = data["blobs"]
    median_single = data["median_single"]
    W, H = data["W"], data["H"]

    # 3. FOND LOCAL
    print("Building local background maps ...")
    bg_green = build_bg_map(green_u8, cfg.bg_grid)
    bg_red   = build_bg_map(red_u8,   cfg.bg_grid)
    bg_blue  = build_bg_map(blue_u8,  cfg.bg_grid)

    green_bg_global = float(np.median(green_u8))
    red_bg_global   = float(np.median(red_u8))
    blue_bg_global  = float(np.median(blue_u8))
    print(f"Mediane globale  vert={green_bg_global:.1f}  rouge={red_bg_global:.1f}  bleu={blue_bg_global:.1f}")

    # 4. CLASSIFICATION THRESHOLD (toujours faite d'abord)
    results = classify_nuclei(
        nuclei, green_u8, red_u8, blue_u8,
        bg_green, bg_red, bg_blue,
        green_bg_global, red_bg_global, blue_bg_global,
        cfg,
    )

    # 5. AUTO-ENTRAÎNEMENT ML (optionnel)
    ml: Optional[MLClassifier] = None
    features: Optional[np.ndarray] = None

    if parsed.self_train:
        model_path = parsed.model_path
        if Path(model_path).exists():
            ml = MLClassifier(model_path)
            print(f"Modele ML charge ({len(ml.models)} classifieurs)")
        else:
            print("Auto-entrainement du Random Forest sur les predictions threshold ...")
            features, feature_names = extract_features(
                nuclei, green_u8, red_u8, blue_u8, white_u8,
                bg_green, bg_red, bg_blue,
                green_bg_global, red_bg_global, blue_bg_global,
                cfg,
            )
            print(f"  -> {features.shape[1]} features pour {features.shape[0]} noyaux")

            y = np.zeros((len(results), 3), dtype=np.int32)
            for i, (_, _, is_g, is_r, is_b, _) in enumerate(results):
                y[i] = [int(is_g), int(is_r), int(is_b)]
            n_pos = y.sum(axis=0)
            print(f"  Labels threshold : G={n_pos[0]}, R={n_pos[1]}, B={n_pos[2]}")

            ml = MLClassifier()
            scores = ml.train(features, y, feature_names)
            print(f"  Score train G={scores['green']:.3f} R={scores['red']:.3f} B={scores['blue']:.3f}")
            ml.save(model_path)

    if ml is not None and features is None:
        features, _ = extract_features(
            nuclei, green_u8, red_u8, blue_u8, white_u8,
            bg_green, bg_red, bg_blue,
            green_bg_global, red_bg_global, blue_bg_global,
            cfg,
        )

    if ml is not None:
        results = classify_nuclei(
            nuclei, green_u8, red_u8, blue_u8,
            bg_green, bg_red, bg_blue,
            green_bg_global, red_bg_global, blue_bg_global,
            cfg, ml, features,
        )
        print("  Classification ML terminee")

    # 5bis. VALIDATION MANUELLE : la vérité terrain confirmée par
    # l'utilisateur (clic sur un noyau dans la GUI) prime sur la prédiction
    # brute pour l'affichage et le rapport. La prédiction brute reste
    # disponible sous raw_results pour l'export CSV (pred_* vs gt_*).
    raw_results = results
    if corrections:
        stats = corrections_summary(corrections)
        print(f"Application de {stats['total']} validation(s) manuelle(s) "
              f"({stats['confirmed']} confirmées, {stats['corrected']} corrigées) ...")
        results = apply_corrections(raw_results, corrections)

    # 6. VISUALISATION
    print("Saving annotated images ...")
    image_paths = save_annotated_images(composite, results, cfg, out_dir)
    print("  -> " + "  ".join(image_paths.keys()))

    # 7. RAPPORT
    total = len(nuclei)
    report = generate_report(cfg, W, H, blobs, median_single, n_ws_split, total, results,
                              correction_stats=corrections_summary(corrections) if corrections else None)
    print(report)
    report_path = out_dir / "cell_report_v5.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Rapport -> {report_path}")

    return {
        "total": total,
        "results": results,
        "raw_results": raw_results,
        "image_paths": image_paths,
        "report": report,
        "report_path": str(report_path),
    }


def main() -> None:
    """Point d'entrée principal (CLI)."""
    cfg, parsed = config_from_args()
    run(cfg, parsed)


if __name__ == "__main__":
    main()
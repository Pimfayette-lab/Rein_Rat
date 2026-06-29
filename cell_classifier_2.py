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
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

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
    for field_name in ("max_dim", "min_area", "max_area", "nucleus_r", "halo_r",
                       "green_thresh", "green_ratio", "red_thresh", "red_stat",
                       "red_ratio", "blue_thresh", "blue_ratio", "bg_grid"):
        val = getattr(parsed, field_name, None)
        if val is not None:
            attr_map = {
                "green_ratio": "green_bg_ratio",
                "red_ratio": "red_bg_ratio",
                "blue_ratio": "blue_bg_ratio",
            }
            setattr(cfg, attr_map.get(field_name, field_name), val)
    return cfg, parsed


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITAIRES IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def to_uint8(arr: np.ndarray, max_dim: Optional[int] = 8192) -> np.ndarray:
    """Normalise un canal en uint8 [0, 255] après downscale optionnel."""
    if max_dim is not None:
        h, w = arr.shape[:2]
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            arr = cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))),
                             interpolation=cv2.INTER_AREA)
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
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
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Charge un CSV de features + labels pour l'entraînement.

    Le CSV doit contenir les colonnes de features (préfixe 'f_') et
    les colonnes 'gt_green', 'gt_red', 'gt_blue' (0/1).
    """
    rows: List[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError(f"Aucune ligne dans {csv_path}")

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
) -> None:
    """Exporte les features + prédictions threshold dans un CSV éditable.

    L'utilisateur peut corriger les colonnes 'gt_green', 'gt_red', 'gt_blue'
    puis lancer --train.
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["cx", "cy", "area", "r_est"] + \
                 [f"f_{n}" for n in feature_names] + \
                 ["pred_green", "pred_red", "pred_blue",
                  "gt_green", "gt_red", "gt_blue"]
        writer.writerow(header)
        for i, (cx, cy, area, r_est) in enumerate(nuclei):
            is_g, is_r, is_b = results[i][2], results[i][3], results[i][4]
            row = [cx, cy, area, r_est] + \
                  [f"{v:.4f}" for v in features[i].tolist()] + \
                  [int(is_g), int(is_r), int(is_b),
                   int(is_g), int(is_r), int(is_b)]
            writer.writerow(row)
    print(f"Features exportees -> {csv_path}")
    print(f"  (Corrigez les colonnes gt_* puis lancez --train {csv_path})")


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
    """Dessine des cercles de rayon proportionnel à r_est."""
    for cx, cy, r_est in cells_r:
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


def save_annotated_images(
    composite: np.ndarray,
    results: List[ClassificationResult],
    cfg: Config,
) -> None:
    """Sauvegarde les images annotées (individuelles + combinée)."""
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

    for fname, cells_xy, col, label in [
        ("annotated_green.png", green_cells, cfg.c_green, "GREEN (AQP2)"),
        ("annotated_red.png",   red_cells,   cfg.c_red,   "RED (AE1)"),
        ("annotated_blue.png",  blue_cells,  cfg.c_blue,  "BLUE"),
    ]:
        vis = composite.copy()
        draw_fixed(vis, cells_xy, col, r=8)
        add_legend(vis, [(label, col, len(cells_xy))])
        cv2.imwrite(fname, vis)

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
    cv2.imwrite("annotated_all.png", vis_all)


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

-- PARAMETRES v5 --
  Canaux  blanc={cfg.ch_white} vert={cfg.ch_green} rouge={cfg.ch_red} bleu={cfg.ch_blue}
  NUCLEUS_R={cfg.nucleus_r}px  HALO_R={cfg.halo_r}px

  GREEN  anneau[{cfg.nucleus_r},{cfg.halo_r}]  seuil>={cfg.green_thresh}  ratio>={cfg.green_bg_ratio}x fond
  RED    {cfg.red_stat}(disque[0,{cfg.nucleus_r}], anneau[{cfg.nucleus_r},{cfg.halo_r}])  seuil>={cfg.red_thresh}  ratio>={cfg.red_bg_ratio}x fond
  BLUE   max(disque[0,{cfg.nucleus_r}], anneau[{cfg.nucleus_r},{cfg.halo_r}])  seuil>={cfg.blue_thresh}  ratio>={cfg.blue_bg_ratio}x fond
  Fond plancher={cfg.bg_floor_frac}x mediane globale
  BG_GRID={cfg.bg_grid}x{cfg.bg_grid}  MAX_DIM={cfg.max_dim}

-- FICHIERS --
  annotated_green.png  annotated_red.png  annotated_blue.png
  annotated_all.png    cell_report_v4.txt
"""


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: Config, parsed: ParsedArgs) -> None:
    """Exécute le pipeline complet : chargement → détection → classification → sorties."""
    # 1. CHARGEMENT
    print(f"Loading CZI from {cfg.czi_path} ...")
    if not Path(cfg.czi_path).exists():
        print(f"ERREUR : fichier introuvable -> {cfg.czi_path}")
        sys.exit(1)
    try:
        img = np.squeeze(imread(cfg.czi_path))
    except Exception as e:
        print(f"ERREUR lors du chargement du CZI : {e}")
        sys.exit(1)
    print(f"Shape brut : {img.shape}  dtype: {img.dtype}")

    if img.ndim < 3:
        print("ERREUR : l'image chargée n'a pas assez de dimensions (attendu >= 3)")
        sys.exit(1)

    white_u8 = to_uint8(img[cfg.ch_white], cfg.max_dim)
    green_u8 = to_uint8(img[cfg.ch_green], cfg.max_dim)
    red_u8   = to_uint8(img[cfg.ch_red],   cfg.max_dim)
    blue_u8  = to_uint8(img[cfg.ch_blue],  cfg.max_dim)
    H, W = white_u8.shape
    print(f"Resolution retenue : {W}x{H} px")

    composite = build_composite(green_u8, red_u8, blue_u8, white_u8)

    # 2. DÉTECTION
    nuclei, n_ws_split, blobs, median_single = detect_nuclei(white_u8, cfg)
    print(f"Noyaux : {len(nuclei)}  (dont {n_ws_split} blobs decoupes par watershed)")

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

    # 6. VISUALISATION
    print("Saving annotated images ...")
    save_annotated_images(composite, results, cfg)
    print("  -> annotated_green.png  annotated_red.png  annotated_blue.png  annotated_all.png")

    # 7. RAPPORT
    total = len(nuclei)
    report = generate_report(cfg, W, H, blobs, median_single, n_ws_split, total, results)
    print(report)
    with open("cell_report_v4.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("Rapport -> cell_report_v4.txt")


def main() -> None:
    """Point d'entrée principal."""
    cfg, parsed = config_from_args()
    run(cfg, parsed)


if __name__ == "__main__":
    main()

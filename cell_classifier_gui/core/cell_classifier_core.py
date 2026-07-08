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
import logging
import pickle
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
from czifile import imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

    # Normalisation robuste (clip percentile) avant conversion en uint8,
    # + gamma, réglables INDIVIDUELLEMENT par canal. Remplace le min/max
    # brut : quelques pixels chauds (saturation capteur, poussière, artefact
    # ponctuel) ne doivent pas à eux seuls écraser tout le reste du signal
    # dans les valeurs basses. *_clip_low/high sont les percentiles (0-100)
    # utilisés comme bornes noir/blanc (contraste) ; *_gamma ajuste la
    # luminosité des tons moyens sans bouger ces bornes (gamma > 1 = plus
    # clair, gamma < 1 = plus sombre, 1.0 = neutre).
    #
    # Ces réglages affectent À LA FOIS l'aperçu visuel ET la classification
    # (les seuils GREEN/RED/BLUE s'appliquent sur ces canaux normalisés) :
    # ajuster le contraste d'un canal peut donc aussi changer le comptage.
    white_clip_low: float = 0.5
    white_clip_high: float = 99.5
    white_gamma: float = 1.0

    green_clip_low: float = 0.5
    green_clip_high: float = 99.5
    green_gamma: float = 1.0

    red_clip_low: float = 0.5
    red_clip_high: float = 99.5
    red_gamma: float = 1.0

    blue_clip_low: float = 0.5
    blue_clip_high: float = 99.5
    blue_gamma: float = 1.0

    # Détection noyaux
    min_area: int = 50
    max_area: int = 80000
    merge_ratio: float = 1.6

    # Méthode de détection des noyaux :
    #  - "watershed" : seuillage (Otsu global ou local) + watershed (défaut, rapide)
    #  - "cellpose"  : segmentation par instance via Cellpose, plus robuste
    #    sur les amas denses mais nécessite le package `cellpose`
    #    (pip install cellpose) et idéalement un GPU CUDA pour rester
    #    rapide sur de grandes images.
    detection_method: str = "watershed"
    # Diamètre moyen (px) attendu par Cellpose ; None = estimation automatique.
    cellpose_diameter: Optional[float] = None
    # Force l'usage du GPU CUDA pour Cellpose. Si True et qu'aucun GPU
    # CUDA n'est réellement disponible pour PyTorch, une erreur explicite
    # est levée plutôt que de retomber silencieusement sur le CPU (ce qui
    # rend l'inférence extrêmement lente sur de grandes images, plusieurs
    # dizaines de minutes voire plus, sans aucun message clair).
    cellpose_gpu: bool = True

    # Seuil de détection local : au lieu d'un unique seuil Otsu global sur
    # tout le canal blanc, calcule un seuil Otsu par tuile (grille bg_grid)
    # puis l'interpole sur l'image (même principe que build_bg_map). Utile
    # seulement si l'éclairage/le fond varie fortement selon la zone de
    # l'image (vignettage, gradient d'illumination) et que le seuil global
    # sur/sous-détecte selon la position. Sans effet si detection_method
    # vaut "cellpose".
    local_threshold: bool = False

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
    for _ch in ("white", "green", "red", "blue"):
        p.add_argument(f"--clip-low-{_ch}", type=float, default=0.5,
                       help=f"Percentile bas (0-100) — contraste canal {_ch}")
        p.add_argument(f"--clip-high-{_ch}", type=float, default=99.5,
                       help=f"Percentile haut (0-100) — contraste canal {_ch}")
        p.add_argument(f"--gamma-{_ch}", type=float, default=1.0,
                       help=f"Gamma (luminosité, 1.0=neutre) — canal {_ch}")
    p.add_argument("--min-area", type=int, default=50, help="Aire minimale d'un blob")
    p.add_argument("--max-area", type=int, default=80000, help="Aire maximale d'un blob")
    p.add_argument("--detection-method", choices=["watershed", "cellpose"], default="watershed",
                   help="Méthode de détection des noyaux")
    p.add_argument("--cellpose-diameter", type=float, default=None,
                   help="Diamètre moyen (px) attendu par Cellpose (vide = auto)")
    p.add_argument("--cellpose-cpu", action="store_true",
                   help="Force Cellpose à tourner sur CPU même si un GPU CUDA est détecté")
    p.add_argument("--local-threshold", action="store_true",
                   help="Seuil Otsu local par tuile au lieu d'un seuil global unique")
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
    p.add_argument("--train", metavar="CSV",
                   help="Entraîne un modèle supervisé (RF) depuis un CSV de features "
                        "exporté par la GUI, avec split train/test, puis quitte.")
    p.add_argument("--test-size", type=float, default=0.25,
                   help="Fraction des données réservée au test (--train uniquement)")
    p.add_argument("--all-rows", action="store_true",
                   help="Avec --train : utilise toutes les lignes du CSV, pas "
                        "seulement celles validées manuellement ('validated'=1)")
    return p


ParsedArgs = argparse.Namespace


def config_from_args(args: Optional[List[str]] = None) -> Tuple[Config, ParsedArgs]:
    """Crée un Config et retourne le namespace argparse complet."""
    cfg = Config()
    parser = build_parser()
    parsed = parser.parse_args(args)
    if parsed.czi_path:
        cfg.czi_path = parsed.czi_path
    field_names = ["max_dim", "min_area", "max_area", "nucleus_r", "halo_r",
                   "green_thresh", "green_ratio", "red_thresh", "red_stat",
                   "red_ratio", "blue_thresh", "blue_ratio", "bg_grid",
                   "detection_method", "cellpose_diameter", "local_threshold"]
    attr_map = {
        "green_ratio": "green_bg_ratio",
        "red_ratio": "red_bg_ratio",
        "blue_ratio": "blue_bg_ratio",
    }
    for _ch in ("white", "green", "red", "blue"):
        for _suffix, _attr_suffix in (("clip_low", "clip_low"), ("clip_high", "clip_high"), ("gamma", "gamma")):
            field_names.append(f"{_suffix}_{_ch}")
            attr_map[f"{_suffix}_{_ch}"] = f"{_ch}_{_attr_suffix}"
    for field_name in field_names:
        val = getattr(parsed, field_name, None)
        if val is not None:
            setattr(cfg, attr_map.get(field_name, field_name), val)
    if getattr(parsed, "cellpose_cpu", False):
        cfg.cellpose_gpu = False
    return cfg, parsed


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITAIRES IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def to_uint8(
    arr: np.ndarray,
    max_dim: Optional[int] = 8192,
    clip_low: float = 0.5,
    clip_high: float = 99.5,
    gamma: float = 1.0,
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

    `gamma` ajuste ENSUITE la luminosité des tons moyens, sans déplacer les
    bornes noir/blanc fixées par clip_low/high (donc indépendant du
    contraste) : gamma > 1 éclaircit, gamma < 1 assombrit, 1.0 = neutre.
    Pratique pour rendre un signal faible plus visible (ou au contraire
    atténuer un fond trop présent) sans re-toucher aux seuils de clip.
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
    if gamma is not None and gamma != 1.0:
        arr = np.power(arr, 1.0 / max(gamma, 1e-3))
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


def build_local_threshold_map(ch_u8: np.ndarray, grid: int = 8) -> np.ndarray:
    """Carte de seuil Otsu LOCAL (par tuile de la grille grid×grid),
    interpolée sur toute l'image — alternative à un unique seuil Otsu
    global pour la détection des noyaux (voir detect_nuclei).

    Un seuil global unique suppose un éclairage/fond homogène sur toute
    l'image. Si ce n'est pas le cas (vignettage, gradient d'illumination,
    zones de densité cellulaire très différentes), il peut sur-détecter
    dans les zones sombres (bruit pris pour du signal) ou sous-détecter
    dans les zones claires (signal réel sous le seuil global). Calculer un
    Otsu par tuile puis interpoler adapte le seuil à chaque zone.

    Le calcul par tuile utilise cv2.threshold (Otsu), qui n'est pas
    vectorisable sur toutes les tuiles à la fois ; la boucle reste peu
    coûteuse car limitée à grid×grid tuiles (ex. 8×8 = 64 itérations),
    pas à des pixels ou des noyaux individuels.
    """
    H, W = ch_u8.shape
    th = max(1, H // grid)
    tw = max(1, W // grid)
    thresh_map = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        y0 = i * th
        y1 = H if i == grid - 1 else (i + 1) * th
        for j in range(grid):
            x0 = j * tw
            x1 = W if j == grid - 1 else (j + 1) * tw
            tile = ch_u8[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            if tile.max() == tile.min():
                # Tuile quasi constante : Otsu n'a pas de sens, on prend
                # un seuil légèrement au-dessus du niveau constant.
                thresh_map[i, j] = float(tile.mean())
                continue
            t, _ = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_map[i, j] = t
    return cv2.resize(thresh_map, (W, H), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
#  ÉCHANTILLONNAGE RADIAL
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _ring_template(r_inner: int, r_outer: int) -> Tuple[np.ndarray, np.ndarray]:
    """Décalages (dy, dx) d'un anneau [r_inner, r_outer] centré en (0, 0).

    Précalculé UNE SEULE FOIS par couple de rayons (mis en cache), puisque
    tous les noyaux d'une même analyse réutilisent exactement les mêmes
    rayons (cfg.nucleus_r / cfg.halo_r). Évite de reconstruire un meshgrid
    et un masque booléen à chaque appel — c'était le principal coût CPU de
    l'échantillonnage radial (des dizaines de milliers de noyaux × plusieurs
    anneaux/canaux chacun).
    """
    ys = np.arange(-r_outer, r_outer + 1)
    xs = np.arange(-r_outer, r_outer + 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    d2 = yy ** 2 + xx ** 2
    ring = (d2 > r_inner ** 2) & (d2 <= r_outer ** 2)
    return yy[ring].astype(np.int32), xx[ring].astype(np.int32)


def sample_ring(ch: np.ndarray, cx: int, cy: int,
                r_inner: int, r_outer: int,
                statistic: str = "mean") -> float:
    """Statistique des pixels dans l'anneau [r_inner, r_outer].

    statistic peut être "mean", "max", "p95", "p90", "std".

    Version vectorisée : les décalages de l'anneau sont précalculés une
    fois (voir _ring_template) puis simplement décalés au centre (cx, cy)
    et utilisés en indexation avancée — pas de meshgrid ni de découpe de
    patch recalculés à chaque noyau.
    """
    dy, dx = _ring_template(r_inner, r_outer)
    H, W = ch.shape
    ys = cy + dy
    xs = cx + dx
    inb = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    if not np.any(inb):
        return 0.0
    vals = ch[ys[inb], xs[inb]]
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

@lru_cache(maxsize=None)
def _radial_bands_template(radii: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Décalages (dy, dx) et index de bande pour un profil radial complet,
    centré en (0, 0). Précalculé une fois par tuple de rayons (mis en
    cache) puis simplement décalé au centre de chaque noyau — voir
    _ring_template pour la même idée appliquée à sample_ring/sample_disk.
    """
    max_r = radii[-1]
    ys = np.arange(-max_r, max_r + 1)
    xs = np.arange(-max_r, max_r + 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dist = np.sqrt((yy.astype(np.float64) ** 2 + xx.astype(np.float64) ** 2))
    radii_arr = np.asarray(radii, dtype=np.float64)
    # bande b telle que radii[b] <= dist < radii[b+1] (même convention que
    # l'ancienne boucle "dists >= r0) & (dists < r1)").
    # NB : le pixel central exact (dist == 0) est exclu, à l'identique de
    # l'ancienne implémentation (qui réutilisait _make_ring_coords avec
    # r_inner=0, dont le masque est strictement `d2 > 0`) — comportement
    # existant préservé pour ne pas changer les features/seuils appris.
    band_idx = np.searchsorted(radii_arr, dist, side="right") - 1
    n_bands = len(radii) - 1
    valid = (band_idx >= 0) & (band_idx < n_bands) & (dist < radii_arr[-1]) & (dist > 0)
    return yy[valid].astype(np.int32), xx[valid].astype(np.int32), band_idx[valid].astype(np.int32)


def _band_stats(band: np.ndarray) -> Tuple[float, float, float, float, float]:
    """(mean, std, median, p25, p75) d'un vecteur 1D, en un seul tri au
    lieu de 3 appels np.percentile/np.median séparés (chacun refait un
    partitionnement). Pour des bandes de quelques dizaines à centaines de
    pixels (cas typique ici), l'overhead d'appel dominé par 3 calls
    numpy distincts est plus coûteux qu'un unique np.sort suivi d'un
    indexage — gain mesuré ~4-5x sur cette étape, elle-même appelée des
    dizaines de fois par noyau (5 bandes × 4 canaux) sur des populations
    de dizaines de milliers de noyaux.

    L'interpolation linéaire reproduit exactement celle par défaut de
    np.percentile (méthode 'linear').
    """
    mean = float(band.mean())
    std = float(band.std())
    sb = np.sort(band)
    n = len(sb)

    def _pct(p: float) -> float:
        idx = (n - 1) * p / 100.0
        lo = int(np.floor(idx))
        hi = int(np.ceil(idx))
        if lo == hi:
            return float(sb[lo])
        return float(sb[lo] + (sb[hi] - sb[lo]) * (idx - lo))

    return mean, std, _pct(50.0), _pct(25.0), _pct(75.0)


def radial_profile(
    ch: np.ndarray, cx: int, cy: int, radii: List[int]
) -> List[float]:
    """Profil radial complet en un seul passage.

    Découpe le disque de rayon radii[-1] en bandes concentriques définies
    par radii, et retourne pour chaque bande : mean, std, median, p25, p75.

    Version vectorisée : le gabarit (décalages + index de bande) est
    précalculé une seule fois par tuple de rayons (voir
    _radial_bands_template) au lieu de reconstruire un meshgrid et de
    recalculer toutes les distances à chaque noyau.
    """
    dy, dx, band_idx = _radial_bands_template(tuple(radii))
    H, W = ch.shape
    ys = cy + dy
    xs = cx + dx
    inb = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    ys = ys[inb]
    xs = xs[inb]
    bidx = band_idx[inb]
    vals = ch[ys, xs].astype(np.float32)

    n_bands = len(radii) - 1
    out: List[float] = []
    for b in range(n_bands):
        band = vals[bidx == b]
        if band.size == 0:
            out.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            out.extend(_band_stats(band))
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


def train_supervised_from_csv(
    csv_path: str,
    model_path: str,
    validated_only: bool = True,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    """Entraîne un vrai modèle supervisé (RF ×3) sur un CSV de features
    exporté depuis la GUI, avec un split train/test — ferme la boucle du
    système de validation manuelle (clic sur les noyaux) déjà en place.

    Contrairement à MLClassifier.train() seul (utilisé par --self-train),
    qui n'évalue que sur les données d'entraînement elles-mêmes (score
    optimiste, ne détecte pas le sur-apprentissage), cette fonction met de
    côté une fraction `test_size` des noyaux JAMAIS vus pendant
    l'entraînement et rapporte le score sur cette fraction : c'est le
    score de généralisation qui compte réellement pour juger si le modèle
    est utilisable sur de nouvelles images.

    validated_only=True (recommandé) : n'entraîne que sur les lignes où la
    vérité terrain a été confirmée manuellement dans la GUI (colonne
    'validated'=1), pas sur les lignes qui recopient simplement la
    prédiction brute (ce qui reviendrait à réapprendre ses propres seuils).

    Retourne un résumé exploitable par la GUI (ou le CLI) :
        {"n_total", "n_train", "n_test",
         "scores_train": {"green":.., "red":.., "blue":..},
         "scores_test":  {"green":.., "red":.., "blue":..},
         "model_path"}

    Lève ValueError si trop peu d'exemples validés sont disponibles pour un
    split fiable (il faut alors valider davantage de noyaux dans la GUI).
    """
    X, y, feature_names = prepare_training_data(csv_path, validated_only=validated_only)
    n = X.shape[0]
    if n < 10:
        raise ValueError(
            f"Trop peu d'exemples ({n}) pour un split train/test fiable. "
            f"Validez davantage de noyaux dans la GUI (clic sur un cercle) "
            f"puis ré-exportez le CSV avant d'entraîner."
        )

    idx = np.arange(n)
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state,
    )
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    ml = MLClassifier()
    ml.feature_names = feature_names
    scores_train: Dict[str, float] = {}
    scores_test: Dict[str, float] = {}

    for i, label in enumerate(["green", "red", "blue"]):
        m = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
        m.fit(X_train, y_train[:, i])
        ml.models[label] = m
        scores_train[label] = float(m.score(X_train, y_train[:, i]))
        scores_test[label] = (
            float(m.score(X_test, y_test[:, i])) if len(X_test) else float("nan")
        )

    ml.save(model_path)

    return {
        "n_total": n,
        "n_train": len(idx_train),
        "n_test": len(idx_test),
        "scores_train": scores_train,
        "scores_test": scores_test,
        "model_path": model_path,
    }


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

    Si cfg.detection_method == "cellpose", délègue entièrement à
    _detect_nuclei_cellpose (segmentation par instance, pas de watershed
    de fusion nécessaire ensuite).
    """
    if cfg.detection_method == "cellpose":
        return _detect_nuclei_cellpose(white_u8, cfg)

    blur = cv2.GaussianBlur(white_u8, (5, 5), 1)
    if cfg.local_threshold:
        thresh_map = build_local_threshold_map(blur, cfg.bg_grid)
        th = ((blur.astype(np.float32) > thresh_map).astype(np.uint8)) * 255
    else:
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


def _detect_nuclei_cellpose(
    white_u8: np.ndarray, cfg: Config
) -> Tuple[List[NucleusRecord], int, List, float]:
    """Détection des noyaux par segmentation d'instance Cellpose, alternative
    au pipeline Otsu + watershed.

    Chantier le plus lourd des méthodes de détection : nécessite le
    package `cellpose` (`pip install cellpose`, idéalement avec un GPU
    CUDA pour rester rapide sur de grandes images), mais c'est en général
    la méthode la plus fiable sur des amas de noyaux très denses/collés,
    là où le watershed a tendance soit à sur-découper soit à fusionner.

    Cellpose sépare déjà les instances individuellement : il n'y a donc
    plus besoin de l'étape watershed de fusion/découpe de blobs -> les
    valeurs retournées pour n_ws_split/blobs/median_single sont
    informatives (compatibles avec le rapport) mais pas utilisées pour
    re-découper quoi que ce soit.

    Si cfg.cellpose_gpu est True (par défaut), le GPU CUDA est exigé
    explicitement : si aucun GPU CUDA n'est réellement accessible à
    PyTorch, on lève une erreur claire plutôt que de retomber
    silencieusement sur le CPU (ce qui rend l'inférence extrêmement
    lente — plusieurs dizaines de minutes voire plus sur une grande
    image — sans qu'on comprenne pourquoi ça semble "bloqué").
    """
    try:
        from cellpose import models
    except ImportError as e:
        raise RuntimeError(
            "detection_method='cellpose' nécessite le package 'cellpose' "
            "(pip install cellpose). Sans GPU CUDA disponible, la "
            "segmentation sera lente sur de grandes images."
        ) from e

    # Avertissement interne bénin de PyTorch (vérifications d'invariants
    # sparse désactivées par défaut sur GPU pour la perf) — sans rapport
    # avec notre code, juste du bruit dans le journal.
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="Sparse invariant checks are implicitly disabled",
        category=UserWarning,
    )

    use_gpu = bool(cfg.cellpose_gpu)
    if use_gpu:
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "cellpose_gpu=True nécessite PyTorch installé avec le "
                "support CUDA (torch.cuda). Le package 'torch' n'est pas "
                "importable du tout ici : réinstalle-le avec la commande "
                "CUDA correspondant à ton GPU depuis "
                "https://pytorch.org/get-started/locally/ , ou repasse en "
                "CPU (décoche 'Forcer GPU CUDA' / --cellpose-cpu)."
            ) from e
        if not torch.cuda.is_available():
            raise RuntimeError(
                "cellpose_gpu=True mais torch.cuda.is_available() est "
                "False : aucun GPU CUDA utilisable par PyTorch n'a été "
                "détecté. Causes fréquentes : (1) c'est la version CPU-only "
                "de torch qui est installée (pip install torch tout court "
                "installe souvent celle-ci) — réinstalle avec la commande "
                "CUDA adaptée depuis https://pytorch.org/get-started/locally/ "
                "; (2) les drivers NVIDIA/CUDA ne sont pas à jour. Sinon, "
                "décoche 'Forcer GPU CUDA' (--cellpose-cpu en CLI) pour "
                "tourner sur CPU (beaucoup plus lent sur une grande image)."
            )
        print(
            f"[cellpose] GPU CUDA détecté : {torch.cuda.get_device_name(0)} "
            "— utilisation forcée du GPU."
        )
    else:
        gpu_hint = ""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_hint = (
                    f" (remarque : un GPU CUDA est bien détecté par PyTorch "
                    f"— {torch.cuda.get_device_name(0)} — mais l'option "
                    f"'cellpose_gpu' / 'Forcer GPU CUDA' est décochée dans "
                    f"la config, donc le GPU n'est PAS utilisé)"
                )
        except ImportError:
            pass
        print(
            "[cellpose] GPU désactivé par configuration : exécution sur "
            f"CPU (lent).{gpu_hint}"
        )

    try:
        # Anciennes versions de cellpose (<3.0) : classe haut niveau
        # models.Cellpose(model_type=..., gpu=...).
        model = models.Cellpose(model_type="nuclei", gpu=use_gpu)
    except (AttributeError, TypeError):
        # Versions récentes de cellpose (>=3.0/4.0) : la classe
        # models.Cellpose a disparu, il ne reste que CellposeModel. Le
        # modèle "nuclei" pré-entraîné dédié a lui aussi été retiré des
        # versions les plus récentes ; on essaie donc "nuclei" puis on
        # retombe sur le modèle par défaut si besoin.
        try:
            model = models.CellposeModel(pretrained_model="nuclei", gpu=use_gpu)
        except Exception:
            model = models.CellposeModel(gpu=use_gpu)

    # Vérifie a posteriori que le modèle a bien atterri sur le GPU quand
    # demandé (certaines versions de cellpose retombent silencieusement
    # sur le CPU si le GPU est indisponible malgré gpu=True).
    if use_gpu:
        model_device = getattr(model, "device", None)
        if model_device is not None and str(model_device) == "cpu":
            raise RuntimeError(
                "cellpose_gpu=True mais le modèle Cellpose a été chargé sur "
                "CPU malgré tout (model.device == 'cpu'). Vérifie "
                "l'installation de cellpose/torch avec support CUDA."
            )

    # Cellpose journalise sa progression via le module `logging` standard
    # (pas via print/tqdm sur stdout) : sans configuration explicite, ce
    # journal ne va nulle part et l'inférence semble "figée" alors qu'elle
    # tourne normalement. On raccroche son logger à stdout (donc au
    # journal de la GUI) pour voir sa progression réelle.
    cellpose_logger = logging.getLogger("cellpose")
    cellpose_logger.setLevel(logging.INFO)
    _stream_handler = logging.StreamHandler(sys.stdout)
    _stream_handler.setFormatter(logging.Formatter("[cellpose] %(message)s"))
    cellpose_logger.addHandler(_stream_handler)
    cellpose_logger.propagate = False

    n_tiles_hint = ""
    try:
        h, w = white_u8.shape[:2]
        # Taille de tuile par défaut de cellpose (~224 px, chevauchement
        # inclus) : donne un ordre de grandeur, pas un compte exact.
        approx_tiles = max(1, (h // 200) * (w // 200))
        n_tiles_hint = f" (~{approx_tiles} tuiles à traiter, ordre de grandeur)"
    except Exception:
        pass
    print(
        f"[cellpose] Lancement de l'inférence sur une image {white_u8.shape[1]}"
        f"x{white_u8.shape[0]}{n_tiles_hint} — peut prendre plusieurs "
        "minutes selon la taille, même sur GPU. Aucune sortie pendant le "
        "calcul ne signifie PAS que ça a planté : un signal de vie "
        "s'affiche ci-dessous toutes les 15s."
    )

    # Battement de cœur : rassure que le process n'a pas gelé, même si
    # cellpose ne logge rien pendant une tuile particulièrement longue.
    _stop_heartbeat = threading.Event()

    def _heartbeat() -> None:
        start = time.time()
        while not _stop_heartbeat.wait(15):
            print(f"[cellpose] ...toujours en cours ({time.time() - start:.0f}s écoulées)")

    _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    _hb_thread.start()

    try:
        masks, _flows, _styles, *_rest = model.eval(
            white_u8,
            diameter=cfg.cellpose_diameter,
            channels=[0, 0],
        )
    finally:
        _stop_heartbeat.set()
        _hb_thread.join(timeout=1)
        cellpose_logger.removeHandler(_stream_handler)

    print("[cellpose] Inférence terminée, extraction des instances…")

    num_labels = int(masks.max())
    print(
        f"[cellpose] {num_labels} labels détectés — extraction vectorisée "
        "des statistiques par label (aires, centroïdes)…"
    )

    # ATTENTION PERF : ne JAMAIS faire `masks == lid` dans une boucle
    # Python sur tous les labels -> ça recompare l'image entière (des
    # dizaines de millions de pixels) à chaque itération, soit un temps
    # quadratique en (nb_pixels x nb_labels). Avec des dizaines/centaines
    # de milliers de labels (cf. avertissement cellpose "more than 65535
    # masks"), ça peut prendre des heures. À la place : un seul passage
    # sur les pixels non nuls, puis agrégation vectorisée via
    # np.bincount (aire, somme des coordonnées) par label.
    flat = masks.ravel()
    nz_idx = np.flatnonzero(flat)
    lbl_nz = flat[nz_idx].astype(np.int64)
    ys_nz, xs_nz = np.unravel_index(nz_idx, masks.shape)

    area_per_label = np.bincount(lbl_nz, minlength=num_labels + 1)
    cx_sum = np.bincount(lbl_nz, weights=xs_nz.astype(np.float64), minlength=num_labels + 1)
    cy_sum = np.bincount(lbl_nz, weights=ys_nz.astype(np.float64), minlength=num_labels + 1)

    valid = (area_per_label >= cfg.min_area) & (area_per_label <= cfg.max_area)
    valid[0] = False  # label 0 = fond, jamais un noyau
    valid_lids = np.nonzero(valid)[0]

    areas_valid = area_per_label[valid_lids]
    cx_valid = (cx_sum[valid_lids] / areas_valid).astype(np.int64)
    cy_valid = (cy_sum[valid_lids] / areas_valid).astype(np.int64)
    r_valid = np.maximum(4, np.sqrt(areas_valid / np.pi).astype(np.int64))

    nuclei: List[NucleusRecord] = [
        (int(cx), int(cy), int(a), int(r))
        for cx, cy, a, r in zip(cx_valid, cy_valid, areas_valid, r_valid)
    ]
    blobs: List = [
        (int(lid), int(a), int(cx), int(cy))
        for lid, a, cx, cy in zip(valid_lids, areas_valid, cx_valid, cy_valid)
    ]
    areas = [int(a) for a in areas_valid]

    median_single = float(np.median(areas)) if areas else 0.0
    n_ws_split = 0  # Cellpose sépare déjà les instances, pas de découpe watershed
    return nuclei, n_ws_split, blobs, median_single


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
  NORM_CLIP (contraste/gamma par canal) :
    Blanc : [{cfg.white_clip_low}, {cfg.white_clip_high}]% gamma={cfg.white_gamma}
    Vert  : [{cfg.green_clip_low}, {cfg.green_clip_high}]% gamma={cfg.green_gamma}
    Rouge : [{cfg.red_clip_low}, {cfg.red_clip_high}]% gamma={cfg.red_gamma}
    Bleu  : [{cfg.blue_clip_low}, {cfg.blue_clip_high}]% gamma={cfg.blue_gamma}

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
    "white_clip_low", "white_clip_high", "white_gamma",
    "green_clip_low", "green_clip_high", "green_gamma",
    "red_clip_low", "red_clip_high", "red_gamma",
    "blue_clip_low", "blue_clip_high", "blue_gamma",
    "min_area", "max_area", "merge_ratio", "nucleus_r", "halo_r",
    "detection_method", "cellpose_diameter", "cellpose_gpu", "local_threshold",
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

    white_u8 = to_uint8(img[cfg.ch_white], cfg.max_dim, cfg.white_clip_low, cfg.white_clip_high, cfg.white_gamma)
    green_u8 = to_uint8(img[cfg.ch_green], cfg.max_dim, cfg.green_clip_low, cfg.green_clip_high, cfg.green_gamma)
    red_u8   = to_uint8(img[cfg.ch_red],   cfg.max_dim, cfg.red_clip_low,   cfg.red_clip_high,   cfg.red_gamma)
    blue_u8  = to_uint8(img[cfg.ch_blue],  cfg.max_dim, cfg.blue_clip_low,  cfg.blue_clip_high,  cfg.blue_gamma)
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
    if parsed.train:
        summary = train_supervised_from_csv(
            parsed.train, parsed.model_path,
            validated_only=not parsed.all_rows,
            test_size=parsed.test_size,
        )
        print(
            f"\nEntraînement termine sur {summary['n_train']}/{summary['n_total']} "
            f"exemples (test: {summary['n_test']}) :\n"
            f"  Score TRAIN  G={summary['scores_train']['green']:.3f}  "
            f"R={summary['scores_train']['red']:.3f}  B={summary['scores_train']['blue']:.3f}\n"
            f"  Score TEST   G={summary['scores_test']['green']:.3f}  "
            f"R={summary['scores_test']['red']:.3f}  B={summary['scores_test']['blue']:.3f}\n"
            f"  (le score TEST est la mesure de généralisation honnête à surveiller)\n"
            f"Modele sauve -> {summary['model_path']}"
        )
        return
    run(cfg, parsed)


if __name__ == "__main__":
    main()
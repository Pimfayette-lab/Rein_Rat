"""
compare_watershed_cellpose.py
──────────────────────────────
Compare précisément les deux méthodes de détection des noyaux (watershed vs
cellpose) SUR LA MÊME IMAGE PRÉTRAITÉE (même canal blanc, même resize, même
contraste), pour trancher objectivement si Cellpose apporte un vrai gain sur
un CZI donné.

Réutilise directement les fonctions de cell_classifier_core.py (même
pipeline exact que la GUI) pour garantir une comparaison strictement
équitable — aucune divergence de prétraitement entre les deux passes.

Sortie :
  - Rapport texte : comptages, chevauchement (appariement par plus proche
    voisin), statistiques d'aire par méthode, temps d'exécution.
  - Image de superposition (compare_overlay.png) : cercles verts = noyau
    détecté par les DEUX méthodes (apparié), cyan = watershed seul,
    magenta = cellpose seul.

Usage (mêmes options que le script principal pour le chargement/contraste,
czi_path est positionnel comme dans le script d'origine ; le canal blanc
utilisé est celui par défaut de Config.ch_white — pas de flag --ch-white
dans le CLI d'origine) :
    python compare_watershed_cellpose.py "mon_fichier.czi" \
        --max-dim 8192 --min-area 50 --max-area 80000 \
        --cellpose-diameter 30 --output-dir ./compare_out

Si --cellpose-cpu n'est pas passé, le GPU CUDA est exigé pour la passe
Cellpose (erreur explicite sinon, comme dans la GUI).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Même structure d'import que gui.py : ce script DOIT être placé dans le
# même dossier que gui.py (cell_classifier_gui/), avec le module core dans
# le sous-dossier core/ (cell_classifier_gui/core/cell_classifier_core.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from core.cell_classifier_core import (
        Config,
        build_parser,
        detect_nuclei,
        to_uint8,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Impossible de trouver 'core.cell_classifier_core'. Ce script doit "
        "être placé dans le MÊME dossier que gui.py (cell_classifier_gui/), "
        "avec cell_classifier_core.py dans son sous-dossier core/ "
        "(cell_classifier_gui/core/cell_classifier_core.py) — exactement "
        "la même disposition que celle attendue par gui.py."
    ) from e
from czifile import imread


def build_config_from_parsed(parsed: argparse.Namespace) -> Config:
    """Reconstruit un Config à partir du namespace argparse, en reprenant
    exactement la même logique que cell_classifier_core.config_from_args
    (dupliquée ici pour pouvoir ajouter nos propres options --output-dir /
    --match-dist au même parseur sans provoquer d'erreur 'unrecognized
    arguments' lors d'un double parsing)."""
    cfg = Config()
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
    return cfg


def match_nuclei(
    a: list, b: list, max_dist: float
) -> tuple[int, int, int, list[tuple[int, int]]]:
    """Apparie deux listes de noyaux (cx, cy, area, r) par plus proche
    voisin mutuel (chaque point ne peut être apparié qu'une fois), sous
    un seuil de distance max_dist (px).

    Retourne (n_matched, n_only_a, n_only_b, pairs) où pairs est la liste
    des indices (i, j) appariés dans a/b.
    """
    if not a or not b:
        return 0, len(a), len(b), []

    try:
        from scipy.spatial import cKDTree
    except ImportError as e:
        raise RuntimeError(
            "Ce script nécessite scipy (pip install scipy) pour "
            "l'appariement par plus proche voisin."
        ) from e

    pts_a = np.array([(n[0], n[1]) for n in a], dtype=np.float64)
    pts_b = np.array([(n[0], n[1]) for n in b], dtype=np.float64)

    tree_b = cKDTree(pts_b)
    dist, idx_b = tree_b.query(pts_a, k=1, distance_upper_bound=max_dist)

    used_b = set()
    pairs = []
    for i, (d, j) in enumerate(zip(dist, idx_b)):
        if np.isfinite(d) and j not in used_b:
            used_b.add(int(j))
            pairs.append((i, int(j)))

    n_matched = len(pairs)
    n_only_a = len(a) - n_matched
    n_only_b = len(b) - n_matched
    return n_matched, n_only_a, n_only_b, pairs


def area_stats(nuclei: list) -> str:
    if not nuclei:
        return "aucun noyau"
    areas = np.array([n[2] for n in nuclei], dtype=np.float64)
    return (
        f"médiane={np.median(areas):.0f}px²  moyenne={areas.mean():.0f}px²  "
        f"min={areas.min():.0f}  max={areas.max():.0f}"
    )


def main() -> None:
    parser = build_parser()
    parser.add_argument(
        "--output-dir", default="./compare_out",
        help="Dossier de sortie pour le rapport et l'image de superposition",
    )
    parser.add_argument(
        "--match-dist", type=float, default=None,
        help="Distance max (px) pour apparier deux noyaux entre les deux "
             "méthodes (défaut : 1.5x le rayon 'nucleus_r')",
    )
    args = parser.parse_args()
    cfg = build_config_from_parsed(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(cfg.czi_path).exists():
        raise FileNotFoundError(f"Fichier introuvable : {cfg.czi_path}")

    print(f"=== Chargement de {cfg.czi_path} ===")
    img = np.squeeze(imread(cfg.czi_path))
    print(f"Shape brut : {img.shape}  dtype: {img.dtype}")
    white_u8 = to_uint8(
        img[cfg.ch_white], cfg.max_dim, cfg.white_clip_low,
        cfg.white_clip_high, cfg.white_gamma,
    )
    H, W = white_u8.shape
    print(f"Resolution retenue : {W}x{H} px (identique pour les deux méthodes)\n")

    match_dist = args.match_dist if args.match_dist is not None else 1.5 * cfg.nucleus_r

    results = {}
    for method in ("watershed", "cellpose"):
        cfg.detection_method = method
        print(f"--- Détection : {method} ---")
        t0 = time.time()
        nuclei, n_ws_split, blobs, median_single = detect_nuclei(white_u8, cfg)
        elapsed = time.time() - t0
        print(f"{method} : {len(nuclei)} noyaux en {elapsed:.1f}s  ({area_stats(nuclei)})\n")
        results[method] = {"nuclei": nuclei, "elapsed": elapsed}

    n_matched, n_ws_only, n_cp_only, pairs = match_nuclei(
        results["watershed"]["nuclei"], results["cellpose"]["nuclei"], match_dist
    )

    n_ws = len(results["watershed"]["nuclei"])
    n_cp = len(results["cellpose"]["nuclei"])

    report_lines = [
        "=== RAPPORT DE COMPARAISON watershed vs cellpose ===",
        f"Fichier         : {cfg.czi_path}",
        f"Résolution      : {W}x{H} px",
        f"Seuil d'appariement : {match_dist:.1f} px",
        "",
        f"Watershed       : {n_ws} noyaux en {results['watershed']['elapsed']:.1f}s",
        f"Cellpose        : {n_cp} noyaux en {results['cellpose']['elapsed']:.1f}s",
        f"Ratio temps     : cellpose {results['cellpose']['elapsed'] / max(results['watershed']['elapsed'], 1e-6):.1f}x plus lent que watershed",
        "",
        f"Appariés (les deux méthodes s'accordent) : {n_matched}",
        f"Watershed seul (cellpose a raté)          : {n_ws_only}",
        f"Cellpose seul (watershed a raté)           : {n_cp_only}",
        f"Accord global : {100 * n_matched / max(n_ws, n_cp, 1):.1f}% "
        f"(par rapport au plus grand des deux comptages)",
        "",
        f"Aires watershed : {area_stats(results['watershed']['nuclei'])}",
        f"Aires cellpose  : {area_stats(results['cellpose']['nuclei'])}",
    ]
    report_txt = "\n".join(report_lines)
    print(report_txt)

    report_path = out_dir / "compare_report.txt"
    report_path.write_text(report_txt, encoding="utf-8")
    print(f"\nRapport texte : {report_path}")

    # ── Image de superposition ──────────────────────────────────────────
    overlay = cv2.cvtColor(white_u8, cv2.COLOR_GRAY2BGR)
    matched_ws_idx = {i for i, _ in pairs}
    matched_cp_idx = {j for _, j in pairs}

    for i, n in enumerate(results["watershed"]["nuclei"]):
        cx, cy, area, r = n
        color = (0, 255, 0) if i in matched_ws_idx else (255, 255, 0)  # vert=apparié, cyan=watershed seul
        cv2.circle(overlay, (cx, cy), max(3, r), color, 2)

    for j, n in enumerate(results["cellpose"]["nuclei"]):
        cx, cy, area, r = n
        if j not in matched_cp_idx:
            cv2.circle(overlay, (cx, cy), max(3, r), (255, 0, 255), 2)  # magenta=cellpose seul

    overlay_path = out_dir / "compare_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"Image de superposition (vert=appariés, cyan=watershed seul, "
          f"magenta=cellpose seul) : {overlay_path}")


if __name__ == "__main__":
    main()
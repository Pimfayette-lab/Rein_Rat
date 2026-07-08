"""
gui.py
──────
Interface graphique (Tkinter) pour cell_classifier_core.

Permet de :
  - choisir le fichier CZI à analyser et le dossier de sortie
  - régler tous les paramètres de détection/classification depuis l'écran
  - sauvegarder/charger des presets de paramètres (JSON)
  - lancer l'analyse en arrière-plan avec suivi du log en direct
  - visualiser un aperçu de l'image annotée finale à la fin du traitement

Lancement :
    python gui.py
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from typing import Tuple
from dataclasses import fields, asdict
from datetime import datetime
from pathlib import Path
from tkinter import (
    Tk, StringVar, IntVar, DoubleVar, BooleanVar, filedialog, messagebox, END,
    Toplevel, Label, Canvas, TclError,
)
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.cell_classifier_core import (  # noqa: E402
    Config, run as run_pipeline,
    load_and_detect, classify_and_render, summarize_counts,
    SLOW_PARAM_NAMES, build_bg_map,
    Correction, find_nearest_nucleus,
    rerender_with_corrections, save_corrections, load_corrections,
    corrections_summary, classification_label,
    extract_features, export_features_csv,
    train_supervised_from_csv,
)

import numpy as np  # noqa: E402  (déjà une dépendance dure de core.cell_classifier_core)
import cv2  # noqa: E402  (déjà une dépendance dure de core.cell_classifier_core)

try:
    from PIL import Image, ImageTk
    # Ce logiciel manipule volontairement des images scientifiques énormes
    # (microscopie/drone), potentiellement bien au-delà de la limite
    # "decompression bomb" par défaut de Pillow (~178 Mpx). Ces images
    # viennent du fichier CZI que l'utilisateur charge lui-même (pas d'un
    # tiers non fiable) : on désactive donc cette protection anti-DoS, sinon
    # PIL lève DecompressionBombError dès qu'on désactive la limite de
    # redimensionnement (max_dim vide) sur une grande image.
    Image.MAX_IMAGE_PIXELS = None
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


APP_TITLE = "Cell Classifier — AQP2 / AE1"
PRESETS_DIR = Path(__file__).resolve().parent / "presets"
PRESETS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Redirection stdout -> file-like objet basé sur une queue thread-safe
# ─────────────────────────────────────────────────────────────────────────────

class QueueWriter:
    """Redirige print()/tqdm vers une queue consommée par le thread GUI."""

    def __init__(self, q: "queue.Queue[str]") -> None:
        self.q = q

    def write(self, text: str) -> None:
        if text:
            self.q.put(text)

    def flush(self) -> None:
        pass


class ToolTip:
    """Info-bulle simple affichée au survol d'un widget après un court délai."""

    def __init__(self, widget, text: str, delay: int = 450, wraplength: int = 340) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self.tip_window: Toplevel | None = None
        self._after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._unschedule()
        self._after_id = self.widget.after(self.delay, self._show)

    def _unschedule(self) -> None:
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip_window = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        try:
            tw.wm_attributes("-topmost", True)
        except Exception:
            pass
        tw.wm_geometry(f"+{x}+{y}")
        Label(
            tw, text=self.text, justify="left", background="#ffffe0",
            foreground="#222222", relief="solid", borderwidth=1,
            wraplength=self.wraplength, font=("TkDefaultFont", 9),
        ).pack(ipadx=6, ipady=4)

    def _hide(self, _event=None) -> None:
        self._unschedule()
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class ZoomPanCanvas(ttk.Frame):
    """Visionneuse d'image avec zoom (molette) et déplacement (glisser-déposer).

    Ne redimensionne jamais l'image entière : seule la portion visible est
    découpée dans l'image source (pleine résolution) puis redimensionnée,
    ce qui reste rapide même sur des images de plusieurs milliers de pixels.
    """

    MIN_SCALE = 0.02
    MAX_SCALE = 12.0

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self._img = None            # PIL Image, pleine résolution (RGB)
        self._img_w = 0
        self._img_h = 0
        self._scale = 1.0           # px canvas par px image
        self._off_x = 0.0           # coord. image du coin haut-gauche visible
        self._off_y = 0.0
        self._drag_start = None
        self._press_screen_pos = None   # pour distinguer un clic d'un glissé
        self._photo = None          # référence gardée (anti garbage-collect)
        self._placeholder = "Aucun aperçu pour le moment."
        self.on_click = None         # callable(img_x: float, img_y: float) -> None
        self.on_hover = None         # callable(img_x: float, img_y: float) -> None
        self._hover_highlight = None     # (cx, cy, r) en coordonnées image, ou None
        self._selected_highlight = None  # idem, surbrillance persistante (dialogue ouvert)

        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="＋", width=3, command=lambda: self._zoom_step(1.3)).pack(side="left", padx=1)
        ttk.Button(toolbar, text="－", width=3, command=lambda: self._zoom_step(1 / 1.3)).pack(side="left", padx=1)
        ttk.Button(toolbar, text="⤢ Ajuster", command=self.reset_view).pack(side="left", padx=(6, 0))
        self.zoom_label = ttk.Label(toolbar, text="", foreground="#666666")
        self.zoom_label.pack(side="right", padx=4)
        ttk.Label(
            toolbar, text="molette = zoom · glisser/barres = déplacer · double-clic = ajuster",
            foreground="#888888", font=("TkDefaultFont", 8),
        ).pack(side="right", padx=8)

        # Grille : canvas (0,0) + scrollbar verticale (0,1) + scrollbar
        # horizontale (1,0), en plus du zoom/déplacement à la souris déjà
        # existant -> permet de naviguer dans l'image sans molette/glisser,
        # utile notamment avec un pavé tactile ou pour un repérage rapide.
        grid_frame = ttk.Frame(self)
        grid_frame.pack(fill="both", expand=True)
        grid_frame.rowconfigure(0, weight=1)
        grid_frame.columnconfigure(0, weight=1)

        self.canvas = Canvas(grid_frame, bg="#101010", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.vbar = ttk.Scrollbar(grid_frame, orient="vertical", command=self._on_vscroll)
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar = ttk.Scrollbar(grid_frame, orient="horizontal", command=self._on_hscroll)
        self.hbar.grid(row=1, column=0, sticky="ew")

        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", lambda _e: self.reset_view())
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda _e: self.clear_hover_highlight())
        # Molette : Windows/macOS envoient <MouseWheel>, Linux envoie Button-4/5.
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_step(1.15, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_step(1 / 1.15, e.x, e.y))

    def set_click_handler(self, fn) -> None:
        """Enregistre fn(img_x: float, img_y: float) appelé sur un simple
        clic (pas un glissé) sur l'image, avec les coordonnées converties en
        pixels de l'image pleine résolution."""
        self.on_click = fn

    def set_hover_handler(self, fn) -> None:
        """Enregistre fn(img_x: float, img_y: float) appelé à chaque
        déplacement de la souris sur l'image (pour la surbrillance au survol,
        gérée côté appelant qui connaît la position des noyaux)."""
        self.on_hover = fn

    def set_hover_highlight(self, cx: float, cy: float, r: float) -> None:
        """Dessine un anneau de surbrillance « survol » autour d'un point
        donné (coordonnées image), sans retoucher à l'image de fond."""
        self._hover_highlight = (cx, cy, r)
        self._redraw_highlights()

    def clear_hover_highlight(self) -> None:
        if self._hover_highlight is not None:
            self._hover_highlight = None
            self._redraw_highlights()

    def set_selected_highlight(self, cx: float, cy: float, r: float) -> None:
        """Anneau de surbrillance « sélectionné », qui reste affiché tant
        qu'une correction est en cours d'édition pour ce noyau."""
        self._selected_highlight = (cx, cy, r)
        self._redraw_highlights()

    def clear_selected_highlight(self) -> None:
        if self._selected_highlight is not None:
            self._selected_highlight = None
            self._redraw_highlights()

    def canvas_to_image_coords(self, cx: float, cy: float) -> Tuple[float, float]:
        """Convertit des coordonnées canvas (event.x, event.y) en coordonnées
        image pleine résolution, compte tenu du zoom/déplacement courant."""
        return (self._off_x + cx / self._scale, self._off_y + cy / self._scale)

    def set_image(self, pil_img, reset_view: bool = False) -> None:
        """Affiche une nouvelle image. Par défaut CONSERVE le zoom/déplacement
        courant (utile en réglage interactif, pour garder le cadrage pendant
        qu'on ajuste les seuils). Utiliser reset_view=True pour recadrer."""
        first_image = self._img is None
        self._img = pil_img
        self._img_w, self._img_h = pil_img.size
        if first_image or reset_view:
            self.reset_view()
        else:
            self._redraw()

    def clear(self) -> None:
        self._img = None
        self._redraw()

    def reset_view(self) -> None:
        if self._img is None:
            self._redraw()
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        scale = min(cw / self._img_w, ch / self._img_h)
        self._scale = min(max(scale, self.MIN_SCALE), self.MAX_SCALE)
        self._off_x = (self._img_w - cw / self._scale) / 2
        self._off_y = (self._img_h - ch / self._scale) / 2
        self._redraw()

    def _on_press(self, event) -> None:
        if self._img is None:
            return
        self._drag_start = (event.x, event.y, self._off_x, self._off_y)
        self._press_screen_pos = (event.x, event.y)

    def _on_drag(self, event) -> None:
        if self._drag_start is None or self._img is None:
            return
        sx, sy, ox, oy = self._drag_start
        self._off_x = ox - (event.x - sx) / self._scale
        self._off_y = oy - (event.y - sy) / self._scale
        self._clamp_offsets()
        self.clear_hover_highlight()
        self._redraw()

    def _on_motion(self, event) -> None:
        if self._img is None or self.on_hover is None or self._drag_start is not None:
            return
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        self.on_hover(img_x, img_y)

    def _on_release(self, event) -> None:
        start = self._press_screen_pos
        self._drag_start = None
        self._press_screen_pos = None
        if start is None or self._img is None or self.on_click is None:
            return
        moved = ((event.x - start[0]) ** 2 + (event.y - start[1]) ** 2) ** 0.5
        if moved > 4:
            return  # c'était un glissé (pan), pas un clic
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        try:
            self.on_click(img_x, img_y)
        except Exception:
            raise

    def _on_wheel(self, event) -> None:
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self._zoom_step(factor, event.x, event.y)

    def _zoom_step(self, factor: float, cx: float | None = None, cy: float | None = None) -> None:
        if self._img is None:
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        if cx is None:
            cx = cw / 2
        if cy is None:
            cy = ch / 2
        img_x = self._off_x + cx / self._scale
        img_y = self._off_y + cy / self._scale
        new_scale = min(max(self._scale * factor, self.MIN_SCALE), self.MAX_SCALE)
        if new_scale == self._scale:
            return
        self._scale = new_scale
        self._off_x = img_x - cx / self._scale
        self._off_y = img_y - cy / self._scale
        self._clamp_offsets()
        self._redraw()

    def _clamp_offsets(self) -> None:
        """Empêche le cadrage de s'éloigner indéfiniment de l'image (utile
        surtout pour que les barres de scroll restent cohérentes)."""
        if self._img is None:
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        view_w = cw / self._scale
        view_h = ch / self._scale
        # Un peu de marge (une demi-vue) pour ne pas bloquer trop tôt quand
        # l'image est plus petite que la fenêtre.
        max_off_x = max(-view_w / 2, self._img_w - view_w / 2)
        max_off_y = max(-view_h / 2, self._img_h - view_h / 2)
        min_off_x = -view_w / 2
        min_off_y = -view_h / 2
        self._off_x = min(max(self._off_x, min_off_x), max_off_x)
        self._off_y = min(max(self._off_y, min_off_y), max_off_y)

    def _on_hscroll(self, *args) -> None:
        if self._img is None:
            return
        cw = max(self.canvas.winfo_width(), 1)
        view_w = cw / self._scale
        action = args[0]
        if action == "moveto":
            frac = float(args[1])
            self._off_x = frac * self._img_w
        elif action == "scroll":
            num = float(args[1])
            what = args[2]
            step = view_w * 0.9 if what == "pages" else max(1.0, view_w * 0.08)
            self._off_x += num * step
        self._clamp_offsets()
        self.clear_hover_highlight()
        self._redraw()

    def _on_vscroll(self, *args) -> None:
        if self._img is None:
            return
        ch = max(self.canvas.winfo_height(), 1)
        view_h = ch / self._scale
        action = args[0]
        if action == "moveto":
            frac = float(args[1])
            self._off_y = frac * self._img_h
        elif action == "scroll":
            num = float(args[1])
            what = args[2]
            step = view_h * 0.9 if what == "pages" else max(1.0, view_h * 0.08)
            self._off_y += num * step
        self._clamp_offsets()
        self.clear_hover_highlight()
        self._redraw()

    def _update_scrollbars(self) -> None:
        """Met à jour la position/taille des poignées de scrollbar pour
        refléter la portion actuellement visible de l'image."""
        if self._img is None or self._img_w <= 0 or self._img_h <= 0:
            self.hbar.set(0.0, 1.0)
            self.vbar.set(0.0, 1.0)
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        view_w = cw / self._scale
        view_h = ch / self._scale
        lo_x = self._off_x / self._img_w
        hi_x = (self._off_x + view_w) / self._img_w
        lo_y = self._off_y / self._img_h
        hi_y = (self._off_y + view_h) / self._img_h
        self.hbar.set(max(0.0, lo_x), min(1.0, hi_x))
        self.vbar.set(max(0.0, lo_y), min(1.0, hi_y))

    def _redraw(self) -> None:
        self.canvas.delete("all")
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        if self._img is None:
            self.canvas.create_text(
                cw / 2, ch / 2, text=self._placeholder, fill="#888888",
                font=("TkDefaultFont", 10),
            )
            self.zoom_label.configure(text="")
            self._update_scrollbars()
            return

        x0, y0 = self._off_x, self._off_y
        x1, y1 = x0 + cw / self._scale, y0 + ch / self._scale
        crop_x0 = max(0, int(x0))
        crop_y0 = max(0, int(y0))
        crop_x1 = min(self._img_w, int(x1) + 1)
        crop_y1 = min(self._img_h, int(y1) + 1)
        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            self.zoom_label.configure(text=f"{self._scale * 100:.0f}%")
            return

        try:
            crop = self._img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            disp_w = max(1, round((crop_x1 - crop_x0) * self._scale))
            disp_h = max(1, round((crop_y1 - crop_y0) * self._scale))
            resample = Image.NEAREST if self._scale > 1 else Image.BILINEAR
            resized = crop.resize((disp_w, disp_h), resample)
            self._photo = ImageTk.PhotoImage(resized)
        except Exception as e:
            self.canvas.create_text(
                cw / 2, ch / 2,
                text=f"Erreur d'affichage de l'aperçu :\n{e}",
                fill="#ff6666", font=("TkDefaultFont", 9), justify="center",
            )
            self.zoom_label.configure(text="")
            return
        px = (crop_x0 - x0) * self._scale
        py = (crop_y0 - y0) * self._scale
        self.canvas.create_image(px, py, anchor="nw", image=self._photo, tags="base_image")
        self.zoom_label.configure(text=f"{self._scale * 100:.0f}%")
        self._redraw_highlights()
        self._update_scrollbars()

    def _redraw_highlights(self) -> None:
        """Redessine UNIQUEMENT les anneaux de surbrillance (survol/sélection)
        par-dessus l'image déjà affichée, sans retoucher/recadrer/redimensionner
        celle-ci -> reste instantané même en survol continu sur une image
        pleine résolution de plusieurs milliers de pixels."""
        self.canvas.delete("highlight")
        if self._img is None:
            return
        if self._hover_highlight is not None:
            self._draw_highlight_ring(self._hover_highlight, color="#ffd400", width=2)
        if self._selected_highlight is not None:
            self._draw_highlight_ring(self._selected_highlight, color="#ff2fb0", width=3)

    def _draw_highlight_ring(self, spec: Tuple[float, float, float], color: str, width: int) -> None:
        cx, cy, r = spec
        sx = (cx - self._off_x) * self._scale
        sy = (cy - self._off_y) * self._scale
        sr = max(5.0, (r + 5) * self._scale)
        self.canvas.create_oval(
            sx - sr, sy - sr, sx + sr, sy + sr,
            outline=color, width=width, tags="highlight",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Définition des champs de paramètres affichés (mappés sur Config)
# ─────────────────────────────────────────────────────────────────────────────

# (attribut Config, label, type, tooltip)
PARAM_GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Canaux", [
        ("ch_white", "Canal noyaux (blanc)", "int"),
        ("ch_green", "Canal vert (AQP2)", "int"),
        ("ch_red", "Canal rouge (AE1)", "int"),
        ("ch_blue", "Canal bleu", "int"),
    ]),
    ("Chargement image", [
        ("max_dim", "Dimension max (px, vide = pas de limite)", "int_opt"),
    ]),
    ("Contraste — Canal blanc (détection)", [
        ("white_clip_low", "Percentile bas (%)", "float"),
        ("white_clip_high", "Percentile haut (%)", "float"),
        ("white_gamma", "Gamma (luminosité)", "float"),
    ]),
    ("Contraste — Canal vert (AQP2)", [
        ("green_clip_low", "Percentile bas (%)", "float"),
        ("green_clip_high", "Percentile haut (%)", "float"),
        ("green_gamma", "Gamma (luminosité)", "float"),
    ]),
    ("Contraste — Canal rouge (AE1)", [
        ("red_clip_low", "Percentile bas (%)", "float"),
        ("red_clip_high", "Percentile haut (%)", "float"),
        ("red_gamma", "Gamma (luminosité)", "float"),
    ]),
    ("Contraste — Canal bleu", [
        ("blue_clip_low", "Percentile bas (%)", "float"),
        ("blue_clip_high", "Percentile haut (%)", "float"),
        ("blue_gamma", "Gamma (luminosité)", "float"),
    ]),
    ("Détection des noyaux", [
        ("detection_method", "Méthode de détection", "choice:watershed,cellpose"),
        ("cellpose_gpu", "Cellpose : forcer GPU CUDA", "bool"),
        ("cellpose_diameter", "Diamètre Cellpose (px, vide = auto)", "float_opt"),
        ("local_threshold", "Seuil Otsu local (au lieu d'un seuil global)", "bool"),
        ("min_area", "Aire minimale (px²)", "int"),
        ("max_area", "Aire maximale (px²)", "int"),
        ("merge_ratio", "Ratio de fusion (watershed)", "float"),
        ("nucleus_r", "Rayon du noyau (px)", "int"),
        ("halo_r", "Rayon externe du halo (px)", "int"),
    ]),
    ("Canal vert (AQP2)", [
        ("green_thresh", "Seuil signal vert", "int"),
        ("green_bg_ratio", "Ratio vert / fond", "float"),
    ]),
    ("Canal rouge (AE1)", [
        ("red_thresh", "Seuil signal rouge", "int"),
        ("red_stat", "Statistique rouge", "choice:mean,p95,p90,max,std"),
        ("red_bg_ratio", "Ratio rouge / fond", "float"),
        ("red_not_greener", "Exiger rouge > vert (disque)", "bool"),
    ]),
    ("Canal bleu", [
        ("blue_thresh", "Seuil signal bleu", "int"),
        ("blue_bg_ratio", "Ratio bleu / fond", "float"),
    ]),
    ("Fond local", [
        ("bg_floor_frac", "Plancher (fraction médiane globale)", "float"),
        ("bg_grid", "Grille de fond (NxN)", "int"),
    ]),
]

# Explication de l'influence de chaque paramètre sur le calcul, affichée en info-bulle.
TOOLTIPS: dict[str, str] = {
    "ch_white": (
        "Index du canal utilisé pour DÉTECTER les noyaux (seuillage Otsu + "
        "watershed). C'est la base de tout : si ce canal est mal choisi, "
        "aucun noyau n'est trouvé correctement et rien d'autre ne peut "
        "fonctionner en aval."
    ),
    "ch_green": (
        "Index du canal utilisé comme signal AQP2 (vert). Change uniquement "
        "quel canal de l'image est lu pour tester le seuil vert — n'affecte "
        "pas la géométrie de détection."
    ),
    "ch_red": (
        "Index du canal utilisé comme signal AE1 (rouge)."
    ),
    "ch_blue": (
        "Index du canal utilisé comme signal bleu (3ᵉ marqueur)."
    ),
    "max_dim": (
        "Redimensionne l'image si sa plus grande dimension dépasse cette "
        "valeur (px), avant toute analyse. Plus petit = traitement plus "
        "rapide et moins de mémoire, mais les petits noyaux proches "
        "peuvent fusionner et le signal fin est lissé. Plus grand (ou vide) "
        "= plus précis mais plus lent et gourmand en RAM."
    ),
    "detection_method": (
        "'watershed' (défaut) : seuillage Otsu + découpe watershed des "
        "blobs fusionnés, rapide. 'cellpose' : segmentation par instance "
        "via Cellpose — plus robuste sur des amas de noyaux très "
        "denses/collés, mais nécessite le package `cellpose` (pip install "
        "cellpose) et idéalement un GPU CUDA (voir 'Forcer GPU CUDA' "
        "ci-dessous), sans quoi c'est extrêmement lent sur une grande "
        "image. À réserver au cas où le comptage sur amas denses est le "
        "principal problème."
    ),
    "cellpose_gpu": (
        "Si coché (défaut), Cellpose EXIGE un GPU CUDA utilisable par "
        "PyTorch : si aucun n'est détecté, une erreur claire est levée "
        "plutôt que de tourner silencieusement (et très lentement) sur "
        "CPU. Décoche uniquement si tu veux volontairement tourner sur "
        "CPU (test, pas de GPU disponible). Sans effet si la méthode de "
        "détection n'est pas 'cellpose'."
    ),
    "cellpose_diameter": (
        "Diamètre moyen (px) des noyaux attendu par Cellpose. Laisser vide "
        "pour une estimation automatique (recommandé au début). Utilisé "
        "uniquement si la méthode de détection est 'cellpose'."
    ),
    "local_threshold": (
        "Si coché, remplace le seuil Otsu global unique par un seuil Otsu "
        "calculé par tuile (grille 'bg_grid') puis interpolé, comme pour la "
        "carte de fond local. Utile seulement si vous observez des zones "
        "sur- ou sous-détectées selon leur position dans l'image "
        "(éclairage non homogène). Sans effet si la méthode de détection "
        "est 'cellpose'."
    ),
    "min_area": (
        "Aire minimale (px²) pour qu'un blob détecté soit considéré comme "
        "un noyau. Trop bas → du bruit ou des artefacts sont comptés comme "
        "cellules. Trop haut → les petits noyaux réels sont ignorés."
    ),
    "max_area": (
        "Aire maximale (px²) au-delà de laquelle un blob est écarté (fusion "
        "de plusieurs cellules non séparable, artefact, débris). Trop bas "
        "→ des amas légitimes de grande taille sont perdus. Trop haut → des "
        "artefacts volumineux sont comptés."
    ),
    "merge_ratio": (
        "Un blob est considéré comme PLUSIEURS noyaux fusionnés dès que son "
        "aire dépasse merge_ratio × aire médiane d'un noyau isolé — il est "
        "alors découpé par watershed. Plus bas → découpe plus agressive "
        "(risque de sur-découper de gros noyaux uniques). Plus haut → moins "
        "de découpes (risque de compter plusieurs cellules collées comme "
        "une seule)."
    ),
    "nucleus_r": (
        "Rayon (px) du disque central utilisé comme 'noyau' pour "
        "l'échantillonnage radial (RED/BLUE le mesurent aussi dans ce "
        "disque) et comme distance minimale entre deux pics lors du "
        "découpage watershed. Trop petit → capte trop peu de signal, "
        "sur-découpage possible. Trop grand → empiète sur le halo "
        "voisin et fusionne des noyaux proches."
    ),
    "halo_r": (
        "Rayon externe (px) de l'anneau de recherche autour du noyau, dans "
        "lequel le signal GREEN/RED/BLUE est échantillonné (membrane / "
        "signal périnucléaire). Trop petit → rate le signal membranaire "
        "réel. Trop grand → capte le signal de cellules voisines (faux "
        "positifs)."
    ),
    "green_thresh": (
        "Seuil d'intensité absolue (0-255) sur l'anneau vert : en dessous, "
        "la cellule n'est jamais classée GREEN, quel que soit le fond "
        "local. Baisser augmente la sensibilité (plus de faux positifs "
        "possibles) ; augmenter la réduit (plus de faux négatifs "
        "possibles)."
    ),
    "green_bg_ratio": (
        "Le signal vert de l'anneau doit aussi être ≥ ce ratio × le fond "
        "local pour être classé GREEN (contraste signal/fond, indépendant "
        "de l'intensité absolue). Augmenter ce ratio exige un contraste "
        "plus net et réduit les faux positifs dans les zones à fond élevé."
    ),
    "red_thresh": (
        "Seuil d'intensité absolue sur le meilleur des deux "
        "(disque, anneau) rouge selon la statistique choisie ci-dessous. "
        "En dessous, jamais classé RED."
    ),
    "red_stat": (
        "Statistique utilisée pour résumer le signal rouge dans chaque "
        "zone (disque et anneau) : 'mean' lisse le bruit mais rate un "
        "signal ponctuel ; 'p95'/'p90' capturent bien un signal "
        "punctiforme (typique AE1) sans être dominés par un seul pixel "
        "brillant ; 'max' est très sensible au bruit ; 'std' mesure "
        "l'hétérogénéité plutôt que l'intensité."
    ),
    "red_bg_ratio": (
        "Ratio minimal signal rouge / fond local exigé en plus du seuil "
        "absolu. Augmenter le rend plus strict sur le contraste par "
        "rapport au fond, utile si le fond varie beaucoup dans l'image."
    ),
    "red_not_greener": (
        "Si coché, une cellule n'est classée RED que si le signal rouge "
        "dépasse aussi le signal vert du disque au même endroit — évite de "
        "classer RED des cellules où c'est en fait la diaphonie du canal "
        "vert qui domine (bleed-through)."
    ),
    "blue_thresh": (
        "Seuil d'intensité absolue sur le meilleur des deux "
        "(disque, anneau) bleu. Mêmes effets que pour le rouge/vert : plus "
        "bas = plus sensible, plus haut = plus strict."
    ),
    "blue_bg_ratio": (
        "Ratio minimal signal bleu / fond local exigé en plus du seuil "
        "absolu, pour la classification BLUE."
    ),
    "bg_floor_frac": (
        "Plancher appliqué au fond local, exprimé en fraction de la "
        "médiane globale du canal (évite un fond local proche de zéro "
        "dans les zones très sombres, qui rendrait n'importe quel signal "
        "faible 'significatif' par rapport à ce fond quasi nul). Augmenter "
        "ce plancher rend la classification plus stricte dans les zones "
        "sombres de l'image."
    ),
    "bg_grid": (
        "Nombre de tuiles (NxN) utilisées pour construire la carte de fond "
        "local, interpolée ensuite sur toute l'image. Grille fine (N "
        "élevé) → fond très localement adapté mais plus sensible au bruit "
        "et aux variations de densité cellulaire. Grille grossière (N "
        "faible) → fond plus lisse et stable mais moins réactif aux "
        "variations locales d'éclairage."
    ),
}

_CHANNEL_TOOLTIP_LABELS = {
    "white": "blanc (détection des noyaux)",
    "green": "vert (AQP2)",
    "red": "rouge (AE1)",
    "blue": "bleu",
}
for _ch, _label in _CHANNEL_TOOLTIP_LABELS.items():
    TOOLTIPS[f"{_ch}_clip_low"] = (
        f"Percentile bas (0-100) utilisé pour normaliser le canal {_label} "
        f"en 0-255, à la place d'un simple minimum brut. Les pixels en "
        f"dessous de ce percentile sont saturés à 0 (noir). Augmenter "
        f"légèrement (ex. 1-2) ignore un peu plus de bruit de fond très "
        f"sombre SUR CE CANAL UNIQUEMENT."
    )
    TOOLTIPS[f"{_ch}_clip_high"] = (
        f"Percentile haut (0-100) utilisé pour normaliser le canal {_label} "
        f"en 0-255, à la place d'un simple maximum brut. Paramètre clé "
        f"pour la robustesse aux pixels chauds (saturation, poussière, "
        f"artefact ponctuel) : un max brut laisse un seul pixel extrême "
        f"écraser tout le signal réel vers le noir -> faux négatifs. "
        f"Baisser cette valeur (ex. 99.0-99.5) sature ces pixels extrêmes "
        f"au lieu de les laisser dicter toute l'échelle, et redonne du "
        f"contraste au signal utile de ce canal. 100 = ancien comportement "
        f"min/max brut."
    )
    TOOLTIPS[f"{_ch}_gamma"] = (
        f"Ajuste la LUMINOSITÉ des tons moyens du canal {_label}, SANS "
        f"changer le contraste (les bornes noir/blanc fixées par les "
        f"percentiles ci-dessus ne bougent pas). > 1 éclaircit (utile pour "
        f"rendre visible un signal faible ou aider le comptage sur un "
        f"canal peu marqué) ; < 1 assombrit (utile pour atténuer un fond "
        f"trop présent) ; 1.0 = neutre. Affecte À LA FOIS l'aperçu visuel "
        f"ET les seuils de classification (GREEN/RED/BLUE), puisqu'ils "
        f"s'appliquent sur ce même canal normalisé."
    )


class CellClassifierGUI:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)

        self.czi_path = StringVar(value="")
        self.output_dir = StringVar(value=str(Path.home() / "cell_classifier_output"))
        self.self_train = BooleanVar(value=False)
        self.model_path = StringVar(value="cell_classifier_model.pkl")

        self.vars: dict[str, object] = {}
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.last_result: dict | None = None

        # ── État du mode réglage interactif (aperçu live) ──────────────────
        self.interactive_mode = BooleanVar(value=True)
        self.cached_data: dict | None = None          # résultat de load_and_detect()
        self.cache_stale: bool = False                 # True si un paramètre "lent" a changé
        self.detect_thread: threading.Thread | None = None
        self.tuning_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.tuning_job_running: bool = False
        self.tuning_pending: bool = False
        self._pending_cfg: Config | None = None
        self._tuning_after_id: str | None = None
        self.detect_status_var = StringVar(
            value="○ Aucune détection en cache. Cliquez sur « Détecter les noyaux »."
        )
        self.tuning_counts_var = StringVar(value="")
        self.tuning_progress_var = StringVar(value="")
        self._tuning_start_time: float | None = None
        self._tuning_tick_id: str | None = None
        self._preview_has_image: bool = False

        # ── État de la validation manuelle (vérité terrain) ────────────────
        self.corrections: dict[int, Correction] = {}
        self.raw_results: list | None = None     # prédictions brutes (avant correction)
        self._last_bg_maps = None
        self._last_bg_globals = None
        self.corrections_status_var = StringVar(value="0 noyau validé manuellement.")
        self.export_status_var = StringVar(value="")
        self._active_dialog: Toplevel | None = None
        self._active_dialog_idx: int | None = None

        # ── État de l'entraînement supervisé (CSV validé, split train/test) ──
        self.train_csv_path = StringVar(value="")
        self.train_model_out_path = StringVar(value="cell_classifier_model.pkl")
        self.train_test_size = DoubleVar(value=0.25)
        self.train_validated_only = BooleanVar(value=True)
        self.train_status_var = StringVar(value="")
        self.train_thread: threading.Thread | None = None
        self.training_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()

        self._build_layout()
        self._wire_param_traces()
        self._install_global_exception_handler()
        self._poll_log_queue()
        self._poll_tuning_queue()
        self._poll_training_queue()

    def _install_global_exception_handler(self) -> None:
        """Par défaut, Tkinter avale silencieusement toute exception levée
        dans un callback (clic de bouton, trace de variable, after()...).
        C'est particulièrement traître quand l'app tourne sans console
        (ex. lancée via pythonw.exe) : une erreur devient totalement
        invisible. On la fait remonter dans le Journal + une popup."""
        def handler(exc_type, exc_value, exc_tb) -> None:
            text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            try:
                self._log("\n--- ERREUR interne non interceptée ---\n" + text + "\n")
            except Exception:
                pass
            try:
                messagebox.showerror(
                    "Erreur interne",
                    "Une erreur inattendue est survenue. Voir le journal pour le détail.")
            except Exception:
                pass
        self.root.report_callback_exception = handler

    # ── Construction de l'UI ────────────────────────────────────────────

    def _build_layout(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        root_pane = ttk.Panedwindow(self.root, orient="horizontal")
        root_pane.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(root_pane)
        right = ttk.Frame(root_pane)
        root_pane.add(left, weight=3)
        root_pane.add(right, weight=2)

        self._build_file_section(left)
        self._build_tuning_section(left)   # mode réglage interactif (aperçu live)
        self._build_validation_section(left)  # validation manuelle (vérité terrain)
        self._build_training_section(left)  # entraînement supervisé (CSV validé, split train/test)
        self._build_actions(left)          # épinglé en bas -> toujours visible
        self._build_params_notebook(left)  # remplit l'espace restant, défilable

        self._build_log_and_preview(right)

    def _build_file_section(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Fichier & sortie")
        box.pack(fill="x", pady=(0, 8))

        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="Fichier CZI :", width=16).pack(side="left")
        ttk.Entry(row, textvariable=self.czi_path).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row, text="Parcourir…", command=self._browse_czi).pack(side="left")

        row2 = ttk.Frame(box)
        row2.pack(fill="x", padx=8, pady=4)
        ttk.Label(row2, text="Dossier sortie :", width=16).pack(side="left")
        ttk.Entry(row2, textvariable=self.output_dir).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row2, text="Parcourir…", command=self._browse_output_dir).pack(side="left")

        row3 = ttk.Frame(box)
        row3.pack(fill="x", padx=8, pady=4)
        ttk.Checkbutton(row3, text="Auto-entraînement ML (Random Forest)",
                         variable=self.self_train).pack(side="left")
        ttk.Label(row3, text="Modèle :").pack(side="left", padx=(16, 4))
        ttk.Entry(row3, textvariable=self.model_path, width=28).pack(side="left")
        ttk.Button(row3, text="…", width=3, command=self._browse_model).pack(side="left", padx=2)

    def _build_validation_section(self, parent: ttk.Frame) -> None:
        """Section « validation manuelle » : cliquer un noyau dans l'aperçu
        pour confirmer/corriger sa classification, sauver/charger cette
        vérité terrain, et l'exporter en CSV pour un futur entraînement."""
        box = ttk.LabelFrame(parent, text="✅ Validation manuelle (vérité terrain)")
        box.pack(fill="x", pady=(0, 8))

        row1 = ttk.Frame(box)
        row1.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Label(
            row1, textvariable=self.corrections_status_var, foreground="#2f6f2f",
            wraplength=520, justify="left",
        ).pack(side="left", fill="x", expand=True)
        ToolTip(
            row1,
            "Cliquez sur un cercle dans l'aperçu (à droite) pour confirmer ou "
            "corriger sa classification. Les noyaux validés priment sur la "
            "prédiction dans l'aperçu ET dans le rapport final, et "
            "constituent une vérité terrain fiable pour un futur "
            "entraînement supervisé (voir « Exporter CSV » ci-dessous).",
        )

        row2 = ttk.Frame(box)
        row2.pack(fill="x", padx=8, pady=(2, 6))
        ttk.Button(row2, text="💾 Sauver les validations…",
                   command=self._save_corrections_dialog).pack(side="left", padx=2)
        ttk.Button(row2, text="📂 Charger des validations…",
                   command=self._load_corrections_dialog).pack(side="left", padx=2)
        ttk.Button(row2, text="🗑 Tout effacer",
                   command=self._clear_corrections).pack(side="left", padx=2)
        self.export_csv_btn = ttk.Button(
            row2, text="📤 Exporter CSV (features + validations)…",
            command=self._on_export_csv_clicked)
        self.export_csv_btn.pack(side="right", padx=2)
        ToolTip(
            self.export_csv_btn,
            "Extrait le vecteur de features complet de chaque noyau et "
            "l'exporte en CSV, avec les colonnes gt_* remplies par vos "
            "validations manuelles (colonne 'validated'=1) plutôt que "
            "simplement recopiées depuis la prédiction. Base de départ pour "
            "entraîner un vrai modèle supervisé (--train). Étape LENTE "
            "(recalcule les features de tous les noyaux).",
        )

        row3 = ttk.Frame(box)
        row3.pack(fill="x", padx=8, pady=(0, 6))
        self.export_progress = ttk.Progressbar(row3, mode="indeterminate", length=140)
        self.export_progress.pack(side="left")
        ttk.Label(
            row3, textvariable=self.export_status_var, foreground="#a55a2a",
            wraplength=460, justify="left",
        ).pack(side="left", padx=8, fill="x", expand=True)

    def _build_training_section(self, parent: ttk.Frame) -> None:
        """Section « entraînement supervisé » : ferme la boucle du système
        de validation manuelle. Prend un CSV exporté (features + gt_*),
        réserve une fraction des noyaux en test (jamais vue à
        l'entraînement) et rapporte le score sur ce test — la vraie mesure
        de généralisation, contrairement au score train de
        l'auto-entraînement (--self-train)."""
        box = ttk.LabelFrame(parent, text="🧠 Entraînement supervisé (CSV validé)")
        box.pack(fill="x", pady=(0, 8))

        row1 = ttk.Frame(box)
        row1.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Label(row1, text="CSV features :", width=16).pack(side="left")
        ttk.Entry(row1, textvariable=self.train_csv_path).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row1, text="Parcourir…", command=self._browse_train_csv).pack(side="left")

        row2 = ttk.Frame(box)
        row2.pack(fill="x", padx=8, pady=2)
        ttk.Label(row2, text="Modèle à sauver :", width=16).pack(side="left")
        ttk.Entry(row2, textvariable=self.train_model_out_path).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row2, text="…", width=3, command=self._browse_train_model_out).pack(side="left")

        row3 = ttk.Frame(box)
        row3.pack(fill="x", padx=8, pady=2)
        validated_chk = ttk.Checkbutton(
            row3, text="Uniquement les noyaux validés manuellement (recommandé)",
            variable=self.train_validated_only,
        )
        validated_chk.pack(side="left")
        ToolTip(
            validated_chk,
            "Si coché, n'entraîne que sur les lignes 'validated'=1 du CSV "
            "(vérité terrain confirmée par un clic dans la GUI). Décoché, "
            "utilise aussi les lignes non validées, qui recopient la "
            "prédiction brute — le modèle réapprendrait alors en partie ses "
            "propres seuils, ce qui fausse l'évaluation.",
        )

        row4 = ttk.Frame(box)
        row4.pack(fill="x", padx=8, pady=2)
        ttk.Label(row4, text="Fraction test :").pack(side="left")
        ttk.Spinbox(
            row4, from_=0.05, to=0.5, increment=0.05, width=6,
            textvariable=self.train_test_size,
        ).pack(side="left", padx=(4, 12))
        ToolTip(
            row4,
            "Fraction des noyaux réservée à l'évaluation (jamais vue "
            "pendant l'entraînement). 0.25 = 25% des noyaux servent "
            "uniquement à mesurer la généralisation.",
        )
        self.train_btn = ttk.Button(
            row4, text="🎯 Entraîner (train/test split)", command=self._on_train_clicked,
        )
        self.train_btn.pack(side="right")

        row5 = ttk.Frame(box)
        row5.pack(fill="x", padx=8, pady=(2, 6))
        self.train_progress = ttk.Progressbar(row5, mode="indeterminate", length=140)
        self.train_progress.pack(side="left")
        ttk.Label(
            row5, textvariable=self.train_status_var, foreground="#2f6f2f",
            wraplength=460, justify="left",
        ).pack(side="left", padx=8, fill="x", expand=True)

    def _build_tuning_section(self, parent: ttk.Frame) -> None:
        """Section « réglage interactif » : détection en cache + aperçu live
        recalculé à chaque changement de seuil, sans recharger le CZI."""
        box = ttk.LabelFrame(parent, text="🎚 Réglage interactif (aperçu live)")
        box.pack(fill="x", pady=(0, 8))

        row1 = ttk.Frame(box)
        row1.pack(fill="x", padx=8, pady=(6, 2))
        self.detect_btn = ttk.Button(
            row1, text="🔎 Détecter les noyaux (charge le CZI)",
            command=self._on_detect_clicked,
        )
        self.detect_btn.pack(side="left")
        self.detect_progress = ttk.Progressbar(row1, mode="indeterminate", length=140)
        self.detect_progress.pack(side="left", padx=8)
        ToolTip(
            self.detect_btn,
            "Charge le fichier CZI et détecte les noyaux avec les paramètres "
            "actuels des groupes « Canaux », « Chargement image » et "
            "« Détection des noyaux ». Étape LENTE (relit le fichier), à "
            "refaire uniquement quand un de ces paramètres change. Une fois "
            "faite, tous les autres réglages (seuils, ratios...) recalculent "
            "l'aperçu instantanément à partir de ce cache.",
        )

        row2 = ttk.Frame(box)
        row2.pack(fill="x", padx=8, pady=(2, 2))
        interactive_chk = ttk.Checkbutton(
            row2, text="Activer l'aperçu réactif (recalcul auto à chaque changement de seuil)",
            variable=self.interactive_mode, command=self._on_interactive_mode_toggle,
        )
        interactive_chk.pack(side="left")
        ToolTip(
            interactive_chk,
            "Quand activé, chaque modification d'un seuil/ratio de "
            "classification (pas les paramètres de détection) relance "
            "automatiquement classify_and_render() sur les données déjà en "
            "cache et met à jour l'aperçu à droite, sans relire le CZI.",
        )

        row3 = ttk.Frame(box)
        row3.pack(fill="x", padx=8, pady=(2, 6))
        ttk.Label(
            row3, textvariable=self.detect_status_var, foreground="#4a6fa5",
            wraplength=520, justify="left",
        ).pack(side="left", fill="x", expand=True)

        row4 = ttk.Frame(box)
        row4.pack(fill="x", padx=8, pady=(0, 6))
        self.tuning_progress = ttk.Progressbar(row4, mode="indeterminate", length=140)
        self.tuning_progress.pack(side="left")
        ttk.Label(
            row4, textvariable=self.tuning_progress_var, foreground="#a55a2a",
            wraplength=460, justify="left",
        ).pack(side="left", padx=8, fill="x", expand=True)

    def _build_params_notebook(self, parent: ttk.Frame) -> None:
        from tkinter import Canvas, Scrollbar

        nb_frame = ttk.LabelFrame(parent, text="Paramètres de l'analyse")
        nb_frame.pack(fill="both", expand=True, pady=(0, 8))

        canvas = Canvas(nb_frame, highlightthickness=0)
        vscroll = ttk.Scrollbar(nb_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        vscroll.pack(side="right", fill="y", pady=4)

        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event) -> None:
            canvas.itemconfig(inner_id, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event) -> None:
            delta = -1 * (event.delta // 120) if event.delta else 0
            canvas.yview_scroll(int(delta), "units")

        def _on_mousewheel_linux(event) -> None:
            canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

        def _bind_wheel(_event=None) -> None:
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel_linux)
            canvas.bind_all("<Button-5>", _on_mousewheel_linux)

        def _unbind_wheel(_event=None) -> None:
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)

        defaults = Config()

        # Grille en 2 colonnes de LabelFrame par groupe
        for i, (group_name, items) in enumerate(PARAM_GROUPS):
            grp = ttk.LabelFrame(inner, text=group_name)
            grp.grid(row=i // 2, column=i % 2, sticky="nsew", padx=6, pady=6)
            inner.columnconfigure(0, weight=1)
            inner.columnconfigure(1, weight=1)

            for attr, label, kind in items:
                self._add_param_row(grp, attr, label, kind, getattr(defaults, attr))

        ttk.Button(inner, text="↺ Réinitialiser les valeurs par défaut",
                   command=self._reset_defaults).grid(
            row=(len(PARAM_GROUPS) // 2) + 1, column=0, columnspan=2, pady=6)

    def _wire_param_traces(self) -> None:
        """Branche une trace sur chaque variable de paramètre pour piloter le
        mode réglage interactif : les paramètres 'lents' (SLOW_PARAM_NAMES)
        invalident le cache de détection, les autres déclenchent un
        recalcul live de la classification/aperçu (si le mode est actif)."""
        for attr, (var, _kind) in self.vars.items():
            var.trace_add("write", lambda *_args, a=attr: self._on_param_var_changed(a))

    def _on_param_var_changed(self, attr: str) -> None:
        if attr in SLOW_PARAM_NAMES:
            if self.cached_data is not None and not self.cache_stale:
                self.cache_stale = True
                self.detect_status_var.set(
                    "⚠ Paramètre de détection modifié depuis la dernière "
                    "détection — cliquez sur « Détecter les noyaux » pour "
                    "actualiser l'aperçu."
                )
            return
        if self.interactive_mode.get():
            self._schedule_tuning_update()

    def _add_param_row(self, parent, attr: str, label: str, kind: str, default) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=3)
        lbl = ttk.Label(row, text=label, width=32, anchor="w")
        lbl.pack(side="left")

        if kind == "bool":
            v = BooleanVar(value=bool(default))
            widget = ttk.Checkbutton(row, variable=v)
            widget.pack(side="left")
        elif kind.startswith("choice:"):
            options = kind.split(":", 1)[1].split(",")
            v = StringVar(value=str(default))
            widget = ttk.Combobox(row, textvariable=v, values=options, width=10,
                                   state="readonly")
            widget.pack(side="left")
        elif kind in ("int_opt", "float_opt"):
            # valeur numérique optionnelle (ex: max_dim=None) -> case vide
            # par défaut plutôt que d'afficher littéralement "None".
            v = StringVar(value="" if default is None else str(default))
            widget = ttk.Entry(row, textvariable=v, width=12)
            widget.pack(side="left")
        elif kind in ("int", "float"):
            v = StringVar(value=str(default))
            widget = ttk.Entry(row, textvariable=v, width=12)
            widget.pack(side="left")
        else:
            v = StringVar(value=str(default))
            widget = ttk.Entry(row, textvariable=v, width=12)
            widget.pack(side="left")

        tip_text = TOOLTIPS.get(attr)
        if tip_text:
            ToolTip(lbl, tip_text)
            ToolTip(widget, tip_text)
            ttk.Label(row, text="ⓘ", foreground="#4a6fa5", cursor="question_arrow").pack(
                side="left", padx=(4, 0))
            info_icon = row.winfo_children()[-1]
            ToolTip(info_icon, tip_text)

        self.vars[attr] = (v, kind)

    def _build_actions(self, parent: ttk.Frame) -> None:
        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.progress.pack(side="bottom", fill="x", pady=(4, 0))

        box = ttk.Frame(parent)
        box.pack(side="bottom", fill="x", pady=(4, 0))

        ttk.Button(box, text="💾 Sauver preset…", command=self._save_preset).pack(side="left", padx=2)
        ttk.Button(box, text="📂 Charger preset…", command=self._load_preset).pack(side="left", padx=2)

        self.run_btn = ttk.Button(box, text="▶ Lancer l'analyse", command=self._on_run)
        self.run_btn.pack(side="right", padx=2)

        self.open_out_btn = ttk.Button(box, text="Ouvrir dossier sortie",
                                        command=self._open_output_dir, state="disabled")
        self.open_out_btn.pack(side="right", padx=2)

    def _build_log_and_preview(self, parent: ttk.Frame) -> None:
        preview_box = ttk.LabelFrame(parent, text="Aperçu (réglage interactif / résultat final)")
        preview_box.pack(fill="both", expand=True, pady=(0, 8))
        ttk.Label(
            preview_box, textvariable=self.tuning_counts_var,
            font=("TkDefaultFont", 9, "bold"), foreground="#2f6f2f",
            wraplength=520, justify="left",
        ).pack(fill="x", padx=6, pady=(4, 0))
        if PIL_AVAILABLE:
            self.preview_view = ZoomPanCanvas(preview_box)
            self.preview_view.pack(fill="both", expand=True, padx=6, pady=6)
            self.preview_view.set_click_handler(self._on_preview_clicked)
            self.preview_view.set_hover_handler(self._on_preview_hover)
            self.preview_label = None
        else:
            self.preview_view = None
            self.preview_label = ttk.Label(
                preview_box, text="Aucun résultat pour le moment.\n(installer Pillow pour l'aperçu image)",
                anchor="center")
            self.preview_label.pack(fill="both", expand=True, padx=6, pady=6)

        log_box = ttk.LabelFrame(parent, text="Journal")
        log_box.pack(fill="both", expand=True)

        from tkinter import Text, Scrollbar
        text_frame = ttk.Frame(log_box)
        text_frame.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_text = Text(text_frame, wrap="word", height=18, state="disabled",
                              bg="#1e1e1e", fg="#d4d4d4", insertbackground="#d4d4d4")
        scrollbar = Scrollbar(text_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # ── Actions fichiers ────────────────────────────────────────────────

    def _browse_czi(self) -> None:
        path = filedialog.askopenfilename(
            title="Choisir un fichier CZI",
            filetypes=[("Fichiers CZI", "*.czi"), ("Tous les fichiers", "*.*")],
        )
        if path:
            self.czi_path.set(path)
            self._invalidate_cache()

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Choisir le dossier de sortie")
        if path:
            self.output_dir.set(path)

    def _browse_model(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Fichier modèle (.pkl)", defaultextension=".pkl",
            filetypes=[("Modèle pickle", "*.pkl"), ("Tous les fichiers", "*.*")],
        )
        if path:
            self.model_path.set(path)

    def _browse_train_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Choisir le CSV de features exporté",
            filetypes=[("CSV", "*.csv"), ("Tous les fichiers", "*.*")],
        )
        if path:
            self.train_csv_path.set(path)

    def _browse_train_model_out(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Fichier modèle à sauver (.pkl)", defaultextension=".pkl",
            filetypes=[("Modèle pickle", "*.pkl"), ("Tous les fichiers", "*.*")],
        )
        if path:
            self.train_model_out_path.set(path)

    def _open_output_dir(self) -> None:
        path = self.output_dir.get()
        if not path or not Path(path).exists():
            return
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    # ── Presets ──────────────────────────────────────────────────────────

    def _collect_config(self) -> Config:
        cfg = Config()
        if self.czi_path.get():
            cfg.czi_path = self.czi_path.get()

        for attr, (var, kind) in self.vars.items():
            raw = var.get()
            try:
                if kind == "bool":
                    value = bool(raw)
                elif kind == "int_opt":
                    value = None if str(raw).strip() == "" else int(raw)
                elif kind == "float_opt":
                    value = None if str(raw).strip() == "" else float(raw)
                elif kind == "int":
                    value = int(raw)
                elif kind == "float":
                    value = float(raw)
                else:
                    value = raw
            except (ValueError, TypeError) as e:
                raise ValueError(f"Valeur invalide pour « {attr} » : {raw!r}") from e
            setattr(cfg, attr, value)
        return cfg

    def _save_preset(self) -> None:
        try:
            cfg = self._collect_config()
        except ValueError as e:
            messagebox.showerror("Paramètre invalide", str(e))
            return
        path = filedialog.asksaveasfilename(
            title="Sauver le preset", initialdir=str(PRESETS_DIR),
            defaultextension=".json", filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        data = asdict(cfg)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        messagebox.showinfo("Preset sauvé", f"Paramètres enregistrés dans :\n{path}")

    def _load_preset(self) -> None:
        path = filedialog.askopenfilename(
            title="Charger un preset", initialdir=str(PRESETS_DIR),
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "czi_path" in data:
            self.czi_path.set(data["czi_path"])

        for attr, (var, kind) in self.vars.items():
            if attr not in data:
                continue
            value = data[attr]
            if kind in ("int_opt", "float_opt") and value is None:
                var.set("")
            else:
                var.set(value)
        self._invalidate_cache()
        messagebox.showinfo("Preset chargé", f"Paramètres chargés depuis :\n{path}")

    def _reset_defaults(self) -> None:
        defaults = Config()
        for attr, (var, kind) in self.vars.items():
            default = getattr(defaults, attr)
            if kind in ("int_opt", "float_opt"):
                var.set("" if default is None else str(default))
            else:
                var.set(default)
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Oublie la détection en cache (fichier/paramètres de détection changés)."""
        self.cached_data = None
        self.cache_stale = False
        if self.corrections:
            self._log(
                f"[Validation manuelle] {len(self.corrections)} validation(s) "
                f"réinitialisée(s) (cache de détection invalidé).\n"
            )
        self.corrections = {}
        self.raw_results = None
        self._last_bg_maps = None
        self._last_bg_globals = None
        self._update_corrections_status()
        self.detect_status_var.set(
            "○ Aucune détection en cache. Cliquez sur « Détecter les noyaux »."
        )

    # ── Lancement de l'analyse ──────────────────────────────────────────

    def _on_run(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showwarning("Analyse en cours", "Une analyse est déjà en cours.")
            return
        if self.detect_thread is not None and self.detect_thread.is_alive():
            messagebox.showwarning(
                "Détection en cours",
                "La détection interactive est en cours ; patientez avant de "
                "lancer l'analyse complète.",
            )
            return

        if not self.czi_path.get() or not Path(self.czi_path.get()).exists():
            messagebox.showerror("Fichier manquant", "Merci de sélectionner un fichier CZI valide.")
            return

        try:
            cfg = self._collect_config()
        except ValueError as e:
            messagebox.showerror("Paramètre invalide", str(e))
            return

        out_dir = self.output_dir.get().strip() or str(Path.home() / "cell_classifier_output")

        import argparse
        parsed = argparse.Namespace(
            self_train=self.self_train.get(),
            model_path=self.model_path.get() or "cell_classifier_model.pkl",
        )

        # Réutilise la détection déjà en cache (mode réglage interactif) si
        # elle est encore valide -> évite de recharger le CZI inutilement.
        cached_arg = self.cached_data if (self.cached_data is not None and not self.cache_stale) else None

        self._clear_log()
        self._log(f"=== Démarrage de l'analyse ===\nFichier : {cfg.czi_path}\nSortie  : {out_dir}\n\n")
        self.run_btn.configure(state="disabled")
        self.detect_btn.configure(state="disabled")
        self.open_out_btn.configure(state="disabled")
        self.progress.start(12)

        self.worker_thread = threading.Thread(
            target=self._run_worker, args=(cfg, parsed, out_dir, cached_arg, dict(self.corrections)),
            daemon=True,
        )
        self.worker_thread.start()

    def _run_worker(self, cfg: Config, parsed, out_dir: str, cached: dict | None = None,
                     corrections: dict | None = None) -> None:
        writer = QueueWriter(self.log_queue)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = writer
        sys.stderr = writer
        try:
            result = run_pipeline(cfg, parsed, output_dir=out_dir, cached=cached, corrections=corrections)
            self.log_queue.put("__DONE__")
            self.log_queue.put(result)
        except Exception:
            self.log_queue.put("\n--- ERREUR ---\n" + traceback.format_exc())
            self.log_queue.put("__ERROR__")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    # ── Log / polling ────────────────────────────────────────────────────

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", END)
        self.log_text.configure(state="disabled")

    def _log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(END, text)
        self.log_text.see(END)
        self.log_text.configure(state="disabled")

    def _poll_log_queue(self) -> None:
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item == "__DONE__":
                    self._on_run_finished(success=True)
                elif item == "__ERROR__":
                    self._on_run_finished(success=False)
                elif isinstance(item, dict):
                    self.last_result = item
                    self._show_preview(item)
                else:
                    self._log(str(item))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _on_run_finished(self, success: bool) -> None:
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self.detect_btn.configure(state="normal")
        self.open_out_btn.configure(state="normal")
        if success:
            self._log("\n=== Analyse terminée avec succès ===\n")
        else:
            self._log("\n=== Analyse interrompue par une erreur ===\n")
            messagebox.showerror("Erreur pendant l'analyse",
                                  "Voir le journal pour le détail de l'erreur.")

    def _format_counts_text(self, total: int, counts: dict) -> str:
        return (
            f"Total: {total}   G:{counts['green']}  R:{counts['red']}  B:{counts['blue']}  "
            f"Non-classés:{counts['unclassified']}   |   "
            f"G+R:{counts['green_red']}  G+B:{counts['green_blue']}  "
            f"R+B:{counts['red_blue']}  G+R+B:{counts['all_three']}"
        )

    def _show_preview(self, result: dict) -> None:
        results_list = result.get("results")
        if results_list is not None:
            counts = summarize_counts(results_list)
            self.tuning_counts_var.set(self._format_counts_text(result.get("total", len(results_list)), counts))

        image_paths = result.get("image_paths", {})
        path = image_paths.get("annotated_all.png")
        if not path or not Path(path).exists():
            return
        if not PIL_AVAILABLE:
            self.preview_label.configure(
                text=f"Résultat écrit dans :\n{path}\n\n(installer Pillow pour l'aperçu image)")
            return
        img = Image.open(path).convert("RGB")
        # Nouvelle analyse complète -> on recadre la vue sur l'image entière.
        self.preview_view.set_image(img, reset_view=True)
        self._preview_has_image = True

    # ── Mode réglage interactif (détection en cache + aperçu live) ────────

    def _on_detect_clicked(self) -> None:
        """Lance load_and_detect() en arrière-plan : charge le CZI et détecte
        les noyaux avec les paramètres actuels. Étape LENTE, à ne relancer
        que quand un paramètre de détection change."""
        self._log("[Réglage interactif] Clic sur « Détecter les noyaux »…\n")
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showwarning(
                "Analyse en cours", "Une analyse complète est déjà en cours.")
            return
        if self.detect_thread is not None and self.detect_thread.is_alive():
            return
        if not self.czi_path.get() or not Path(self.czi_path.get()).exists():
            messagebox.showerror("Fichier manquant", "Merci de sélectionner un fichier CZI valide.")
            return
        try:
            cfg = self._collect_config()
        except ValueError as e:
            messagebox.showerror("Paramètre invalide", str(e))
            return

        self.detect_btn.configure(state="disabled")
        self.run_btn.configure(state="disabled")
        self.detect_progress.start(12)
        self.detect_status_var.set("⏳ Chargement du CZI et détection des noyaux en cours…")

        self.detect_thread = threading.Thread(
            target=self._detect_worker, args=(cfg,), daemon=True)
        self.detect_thread.start()

    def _detect_worker(self, cfg: Config) -> None:
        writer = QueueWriter(self.log_queue)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = writer
        sys.stderr = writer
        try:
            data = load_and_detect(cfg)
            self.tuning_queue.put(("detect_ok", data))
        except Exception:
            self.tuning_queue.put(("detect_error", traceback.format_exc()))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _on_interactive_mode_toggle(self) -> None:
        if self.interactive_mode.get() and self.cached_data is not None and not self.cache_stale:
            self._schedule_tuning_update(immediate=True)

    def _schedule_tuning_update(self, immediate: bool = False) -> None:
        """Débounce : regroupe les changements rapprochés (ex. frappe clavier)
        en un seul recalcul, ~250 ms après la dernière modification."""
        if self._tuning_after_id is not None:
            try:
                self.root.after_cancel(self._tuning_after_id)
            except Exception:
                pass
        delay = 0 if immediate else 250
        self._tuning_after_id = self.root.after(delay, self._run_tuning_update)

    def _run_tuning_update(self) -> None:
        self._tuning_after_id = None
        if self.cached_data is None or self.cache_stale:
            return
        try:
            cfg = self._collect_config()
        except ValueError:
            # Valeur invalide/incomplète pendant la frappe : on attend la
            # prochaine modification valide plutôt que de planter.
            return
        if self.tuning_job_running:
            self.tuning_pending = True
            self._pending_cfg = cfg
            return
        self._start_tuning_job(cfg)

    def _start_tuning_job(self, cfg: Config) -> None:
        self.tuning_job_running = True
        self._start_tuning_progress()
        cached_snapshot = self.cached_data  # référence figée, jamais mutée en place
        corrections_snapshot = dict(self.corrections)  # idem : jamais mutée en place
        threading.Thread(
            target=self._tuning_worker, args=(cfg, cached_snapshot, corrections_snapshot), daemon=True,
        ).start()

    def _tuning_worker(self, cfg: Config, cached: dict, corrections: dict) -> None:
        writer = QueueWriter(self.log_queue)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = writer
        sys.stderr = writer
        try:
            result = classify_and_render(cfg, cached, corrections=corrections)
            self.tuning_queue.put(("classify_ok", result))
        except Exception:
            self.tuning_queue.put(("classify_error", traceback.format_exc()))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    # ── Indicateur de progression du recalcul (réglage interactif) ────────

    def _start_tuning_progress(self) -> None:
        """Démarre (ou laisse filer si déjà en cours) la barre de progression
        + le chrono affiché pendant le recalcul de la classification/aperçu."""
        self._tuning_start_time = time.time()
        self._log("[Réglage interactif] Recalcul de la classification…\n")
        if self._tuning_tick_id is None:
            self.tuning_progress.start(12)
            self._tick_tuning_progress()

    def _tick_tuning_progress(self) -> None:
        if not self.tuning_job_running:
            self._tuning_tick_id = None
            return
        elapsed = time.time() - (self._tuning_start_time or time.time())
        self.tuning_progress_var.set(f"⏳ Recalcul en cours… ({elapsed:.1f} s)")
        self._tuning_tick_id = self.root.after(150, self._tick_tuning_progress)

    def _stop_tuning_progress(self, ok: bool, extra: str = "") -> None:
        elapsed = time.time() - (self._tuning_start_time or time.time())
        if self._tuning_tick_id is not None:
            try:
                self.root.after_cancel(self._tuning_tick_id)
            except Exception:
                pass
            self._tuning_tick_id = None
        self.tuning_progress.stop()
        if ok:
            self.tuning_progress_var.set(f"✓ Aperçu à jour ({elapsed:.1f} s){(' — ' + extra) if extra else ''}")
            self._log(f"[Réglage interactif] Recalcul terminé en {elapsed:.2f} s{(' — ' + extra) if extra else ''}.\n")
        else:
            self.tuning_progress_var.set(f"❌ Échec du recalcul ({elapsed:.1f} s) — voir le journal.")

    def _apply_tuning_result(self, result: dict) -> None:
        counts = result["counts"]
        total = result["total"]
        self.tuning_counts_var.set(self._format_counts_text(total, counts))

        # "raw_results" n'est présent que sur un vrai recalcul de
        # classification (classify_and_render) ; un simple rerender après
        # correction (rerender_with_corrections) ne le fournit pas et ne
        # doit pas écraser les prédictions brutes déjà connues.
        if "raw_results" in result:
            self.raw_results = result["raw_results"]
        if "bg_maps" in result:
            self._last_bg_maps = result["bg_maps"]
        if "bg_globals" in result:
            self._last_bg_globals = result["bg_globals"]

        vis = result.get("visuals", {}).get("annotated_all.png")
        if vis is None:
            return
        if not PIL_AVAILABLE:
            self.preview_label.configure(text="(installer Pillow pour l'aperçu image)")
            return
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        # reset_view=False : on garde le cadrage/zoom courant pendant le
        # réglage interactif, pour pouvoir observer une zone précise en
        # continu pendant qu'on ajuste les seuils.
        reset = not self._preview_has_image
        self.preview_view.set_image(img, reset_view=reset)
        self._preview_has_image = True

    def _poll_tuning_queue(self) -> None:
        # IMPORTANT : la replanification (root.after en bas) doit TOUJOURS
        # avoir lieu, même si le traitement d'un item plante, sinon le
        # polling s'arrête silencieusement pour le reste de la session
        # (particulièrement invisible si l'app tourne sans console, ex.
        # lancée via pythonw.exe sur Windows).
        try:
            while True:
                try:
                    kind, payload = self.tuning_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._handle_tuning_item(kind, payload)
                except Exception:
                    self._log(
                        "\n--- ERREUR interne (traitement du résultat) ---\n"
                        + traceback.format_exc() + "\n"
                    )
        finally:
            self.root.after(120, self._poll_tuning_queue)

    def _handle_tuning_item(self, kind: str, payload) -> None:
        if kind == "detect_ok":
            self.cached_data = payload
            self.cache_stale = False
            n = len(payload.get("nuclei", []))
            self.detect_status_var.set(
                f"● Détection en cache : {n} noyaux détectés — "
                f"réglages réactifs actifs."
            )
            self._log(f"[Réglage interactif] Détection terminée : {n} noyaux en cache.\n")
            if self.corrections:
                self._log(
                    f"[Validation manuelle] {len(self.corrections)} validation(s) "
                    f"réinitialisée(s) (nouvelle détection — les index de noyaux "
                    f"ne correspondent plus). Rechargez un fichier de validations "
                    f"si besoin.\n"
                )
            self.corrections = {}
            self.raw_results = None
            self._last_bg_maps = None
            self._last_bg_globals = None
            self._update_corrections_status()
            self.detect_btn.configure(state="normal")
            self.run_btn.configure(state="normal")
            self.detect_progress.stop()
            if self.interactive_mode.get():
                self._schedule_tuning_update(immediate=True)

        elif kind == "detect_error":
            self.detect_btn.configure(state="normal")
            self.run_btn.configure(state="normal")
            self.detect_progress.stop()
            self.detect_status_var.set("❌ Échec de la détection — voir le journal.")
            self._log("\n--- ERREUR (détection interactive) ---\n" + str(payload) + "\n")
            messagebox.showerror(
                "Erreur de détection",
                "La détection a échoué. Voir le journal pour le détail.")

        elif kind == "classify_ok":
            self.tuning_job_running = False
            total = payload.get("total")
            extra = f"{total} noyaux classés" if total is not None else ""
            self._stop_tuning_progress(ok=True, extra=extra)
            self._apply_tuning_result(payload)
            if self.tuning_pending:
                self.tuning_pending = False
                pending_cfg = self._pending_cfg
                self._pending_cfg = None
                if pending_cfg is not None:
                    self._start_tuning_job(pending_cfg)

        elif kind == "classify_error":
            self.tuning_job_running = False
            self._stop_tuning_progress(ok=False)
            self._log("\n--- ERREUR (aperçu live) ---\n" + str(payload) + "\n")

        elif kind == "export_ok":
            self.export_csv_btn.configure(state="normal")
            self.export_progress.stop()
            self.export_status_var.set(
                f"✓ Export terminé en {payload['elapsed']:.1f} s — {payload['n']} noyaux.")
            self._log(
                f"[Export CSV] Terminé en {payload['elapsed']:.2f} s -> {payload['path']}\n")
            messagebox.showinfo(
                "Export terminé", f"Features exportées vers :\n{payload['path']}")

        elif kind == "export_error":
            self.export_csv_btn.configure(state="normal")
            self.export_progress.stop()
            self.export_status_var.set("❌ Échec de l'export — voir le journal.")
            self._log("\n--- ERREUR (export CSV) ---\n" + str(payload) + "\n")
            messagebox.showerror(
                "Erreur d'export", "L'export a échoué. Voir le journal pour le détail.")

    # ── Entraînement supervisé (CSV validé, split train/test) ────────────

    def _on_train_clicked(self) -> None:
        if self.train_thread is not None and self.train_thread.is_alive():
            messagebox.showwarning("Entraînement en cours", "Un entraînement est déjà en cours.")
            return
        csv_path = self.train_csv_path.get().strip()
        if not csv_path:
            messagebox.showwarning(
                "CSV manquant",
                "Choisissez d'abord un CSV de features (bouton « 📤 Exporter "
                "CSV » de la section Validation manuelle, ou un export déjà "
                "existant).",
            )
            return
        if not Path(csv_path).exists():
            messagebox.showerror("Fichier introuvable", f"Introuvable :\n{csv_path}")
            return
        model_out = self.train_model_out_path.get().strip()
        if not model_out:
            messagebox.showwarning("Chemin manquant", "Indiquez où sauver le modèle entraîné.")
            return
        try:
            test_size = float(self.train_test_size.get())
        except (TclError, ValueError):
            messagebox.showerror("Valeur invalide", "La fraction de test doit être un nombre.")
            return
        if not (0.05 <= test_size <= 0.5):
            messagebox.showerror("Valeur invalide", "La fraction de test doit être entre 0.05 et 0.5.")
            return

        validated_only = self.train_validated_only.get()
        self.train_btn.configure(state="disabled")
        self.train_progress.start(12)
        self.train_status_var.set("⏳ Entraînement en cours…")
        self._log(
            f"[Entraînement] Démarrage sur {csv_path} "
            f"(validated_only={validated_only}, test_size={test_size})…\n"
        )
        self.train_thread = threading.Thread(
            target=self._train_worker,
            args=(csv_path, model_out, validated_only, test_size),
            daemon=True,
        )
        self.train_thread.start()

    def _train_worker(
        self, csv_path: str, model_out: str, validated_only: bool, test_size: float,
    ) -> None:
        writer = QueueWriter(self.log_queue)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = writer
        sys.stderr = writer
        t0 = time.time()
        try:
            summary = train_supervised_from_csv(
                csv_path, model_out,
                validated_only=validated_only, test_size=test_size,
            )
            summary["elapsed"] = time.time() - t0
            self.training_queue.put(("train_ok", summary))
        except Exception:
            self.training_queue.put(("train_error", traceback.format_exc()))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _poll_training_queue(self) -> None:
        try:
            while True:
                try:
                    kind, payload = self.training_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._handle_training_item(kind, payload)
                except Exception:
                    self._log(
                        "\n--- ERREUR interne (traitement du résultat d'entraînement) ---\n"
                        + traceback.format_exc() + "\n"
                    )
        finally:
            self.root.after(150, self._poll_training_queue)

    def _handle_training_item(self, kind: str, payload) -> None:
        if kind == "train_ok":
            self.train_btn.configure(state="normal")
            self.train_progress.stop()
            st = payload["scores_train"]
            te = payload["scores_test"]
            self.train_status_var.set(
                f"✓ Entraîné en {payload['elapsed']:.1f} s sur "
                f"{payload['n_train']}/{payload['n_total']} noyaux "
                f"(test: {payload['n_test']}) — "
                f"TEST G={te['green']:.2f} R={te['red']:.2f} B={te['blue']:.2f}"
            )
            self._log(
                f"[Entraînement] Terminé en {payload['elapsed']:.2f} s\n"
                f"  Score TRAIN  G={st['green']:.3f}  R={st['red']:.3f}  B={st['blue']:.3f}\n"
                f"  Score TEST   G={te['green']:.3f}  R={te['red']:.3f}  B={te['blue']:.3f}\n"
                f"  (le score TEST — jamais vu à l'entraînement — est la mesure "
                f"de généralisation à surveiller, pas le score train)\n"
                f"  Modèle sauvé -> {payload['model_path']}\n"
            )
            messagebox.showinfo(
                "Entraînement terminé",
                f"{payload['n_train']} noyaux d'entraînement / "
                f"{payload['n_test']} de test.\n\n"
                f"Score TEST (généralisation) :\n"
                f"  Vert  : {te['green']:.3f}\n"
                f"  Rouge : {te['red']:.3f}\n"
                f"  Bleu  : {te['blue']:.3f}\n\n"
                f"Modèle sauvé dans :\n{payload['model_path']}",
            )
        elif kind == "train_error":
            self.train_btn.configure(state="normal")
            self.train_progress.stop()
            self.train_status_var.set("❌ Échec de l'entraînement — voir le journal.")
            self._log("\n--- ERREUR (entraînement supervisé) ---\n" + str(payload) + "\n")
            messagebox.showerror(
                "Erreur d'entraînement",
                "L'entraînement a échoué. Voir le journal pour le détail.",
            )


    # ── Validation manuelle (clic sur un noyau dans l'aperçu) ────────────

    def _on_preview_hover(self, img_x: float, img_y: float) -> None:
        """Surbrillance au survol : met en évidence le noyau sous la souris,
        avec la même logique de sélection que le clic (gère les noyaux
        imbriqués/qui se chevauchent)."""
        if self.raw_results is None or self.preview_view is None:
            return
        try:
            cfg = self._collect_config()
        except ValueError:
            return
        max_dist = max(float(cfg.halo_r), 20.0)
        idx = find_nearest_nucleus(self.raw_results, img_x, img_y, max_dist=max_dist)
        if idx is None:
            self.preview_view.clear_hover_highlight()
            return
        cx, cy, _g, _r, _b, r_est = self.raw_results[idx]
        self.preview_view.set_hover_highlight(cx, cy, r_est)

    def _on_preview_clicked(self, img_x: float, img_y: float) -> None:
        """Appelé par ZoomPanCanvas sur un simple clic (pas un glissé) dans
        l'aperçu, avec les coordonnées converties en pixels image."""
        if self.raw_results is None:
            self._log(
                "[Validation manuelle] Aucune classification en mémoire — "
                "lancez d'abord une détection (« Détecter les noyaux »).\n"
            )
            return
        try:
            cfg = self._collect_config()
        except ValueError:
            return
        max_dist = max(float(cfg.halo_r), 20.0)
        idx = find_nearest_nucleus(self.raw_results, img_x, img_y, max_dist=max_dist)
        if idx is None:
            return  # clic dans le vide : pas de noyau à proximité, on ignore
        self._open_correction_dialog(idx)

    def _open_correction_dialog(self, idx: int) -> None:
        if idx == self._active_dialog_idx and self._active_dialog is not None:
            # Déjà ouvert sur ce noyau -> on ramène juste la fenêtre au premier plan.
            try:
                self._active_dialog.lift()
                self._active_dialog.focus_force()
            except Exception:
                pass
            return

        # Un clic sur un AUTRE noyau pendant qu'une fenêtre de validation est
        # déjà ouverte doit fermer celle-ci (comme un « Annuler ») et ouvrir
        # la nouvelle -> la fenêtre n'est donc jamais modale (pas de grab_set).
        self._close_active_dialog()

        cx, cy, g, r, b, r_est = self.raw_results[idx]
        pred_label = classification_label(g, r, b)

        dlg = Toplevel(self.root)
        dlg.title(f"Noyau #{idx} — validation")
        dlg.transient(self.root)
        dlg.resizable(False, False)
        self._active_dialog = dlg
        self._active_dialog_idx = idx
        if self.preview_view is not None:
            self.preview_view.set_selected_highlight(cx, cy, r_est)

        def close_dialog() -> None:
            self._active_dialog = None
            self._active_dialog_idx = None
            if self.preview_view is not None:
                self.preview_view.clear_selected_highlight()
            try:
                dlg.destroy()
            except Exception:
                pass

        body = ttk.Frame(dlg, padding=14)
        body.pack(fill="both", expand=True)

        def render_step1() -> None:
            for w in body.winfo_children():
                w.destroy()
            ttk.Label(
                body, text=f"Noyau #{idx}   —   position ({cx}, {cy})   —   rayon ≈ {r_est}px",
                font=("TkDefaultFont", 9, "bold"),
            ).pack(anchor="w")
            ttk.Label(body, text=f"Prédiction du modèle : {pred_label}",
                      foreground="#333333").pack(anchor="w", pady=(4, 0))

            current = self.corrections.get(idx)
            if current is not None:
                state_txt = "corrigée" if current.corrected else "confirmée"
                ttk.Label(
                    body,
                    text=f"✓ Déjà validée le {current.validated_at[:19].replace('T', ' ')} "
                         f"({state_txt}) : {current.label()}",
                    foreground="#2f6f2f",
                ).pack(anchor="w", pady=(2, 0))

            ttk.Label(body, text="Cette prédiction est-elle correcte ?",
                      font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(12, 6))

            btn_row = ttk.Frame(body)
            btn_row.pack(fill="x")

            def confirm_correct() -> None:
                self.corrections[idx] = Correction(
                    gt_green=g, gt_red=r, gt_blue=b,
                    was_prediction_correct=True, corrected=False,
                    validated_at=datetime.now().isoformat(timespec="seconds"),
                )
                close_dialog()
                self._after_correction_added(idx, pred_label, pred_label, confirmed=True)

            ttk.Button(btn_row, text="✓ Correcte", command=confirm_correct).pack(side="left", padx=4)
            ttk.Button(btn_row, text="✗ Incorrecte", command=render_step2).pack(side="left", padx=4)
            ttk.Button(btn_row, text="Annuler", command=close_dialog).pack(side="right", padx=4)
            self._reposition_dialog(dlg)

        def render_step2() -> None:
            for w in body.winfo_children():
                w.destroy()
            ttk.Label(
                body, text=f"Noyau #{idx}   —   position ({cx}, {cy})   —   rayon ≈ {r_est}px",
                font=("TkDefaultFont", 9, "bold"),
            ).pack(anchor="w")
            ttk.Label(body, text=f"Prédiction du modèle : {pred_label}",
                      foreground="#333333").pack(anchor="w", pady=(2, 8))
            ttk.Label(body, text="Quelle est la classification correcte ?",
                      font=("TkDefaultFont", 9, "bold")).pack(anchor="w")

            current = self.corrections.get(idx)
            seed_g = current.gt_green if current is not None else g
            seed_r = current.gt_red if current is not None else r
            seed_b = current.gt_blue if current is not None else b
            var_g = BooleanVar(value=seed_g)
            var_r = BooleanVar(value=seed_r)
            var_b = BooleanVar(value=seed_b)

            chk_row = ttk.Frame(body)
            chk_row.pack(anchor="w", pady=(6, 2))
            ttk.Checkbutton(chk_row, text="GREEN (AQP2)", variable=var_g).pack(side="left", padx=4)
            ttk.Checkbutton(chk_row, text="RED (AE1)", variable=var_r).pack(side="left", padx=4)
            ttk.Checkbutton(chk_row, text="BLUE", variable=var_b).pack(side="left", padx=4)
            ttk.Label(
                body, text="(aucune case cochée = noyau non classé)",
                foreground="#888888", font=("TkDefaultFont", 8),
            ).pack(anchor="w", pady=(0, 8))

            btn_row = ttk.Frame(body)
            btn_row.pack(fill="x")

            def submit() -> None:
                new_g, new_r, new_b = var_g.get(), var_r.get(), var_b.get()
                self.corrections[idx] = Correction(
                    gt_green=new_g, gt_red=new_r, gt_blue=new_b,
                    was_prediction_correct=False,
                    corrected=(new_g, new_r, new_b) != (g, r, b),
                    validated_at=datetime.now().isoformat(timespec="seconds"),
                )
                new_label = classification_label(new_g, new_r, new_b)
                close_dialog()
                self._after_correction_added(idx, pred_label, new_label, confirmed=False)

            ttk.Button(btn_row, text="✓ Valider la correction", command=submit).pack(side="left", padx=4)
            ttk.Button(btn_row, text="← Retour", command=render_step1).pack(side="left", padx=4)
            ttk.Button(btn_row, text="Annuler", command=close_dialog).pack(side="right", padx=4)
            self._reposition_dialog(dlg)

        dlg.protocol("WM_DELETE_WINDOW", close_dialog)
        dlg.bind("<Escape>", lambda _e: close_dialog())
        render_step1()

    def _close_active_dialog(self) -> None:
        """Ferme la fenêtre de validation actuellement ouverte, le cas
        échéant (comme si l'utilisateur avait cliqué « Annuler »)."""
        if self._active_dialog is not None:
            try:
                self._active_dialog.destroy()
            except Exception:
                pass
        self._active_dialog = None
        self._active_dialog_idx = None
        if self.preview_view is not None:
            self.preview_view.clear_selected_highlight()

    def _reposition_dialog(self, dlg) -> None:
        """Place la fenêtre de validation par-dessus le cadre de l'aperçu
        (à droite de l'appli), plutôt qu'en haut à gauche de l'écran."""
        if self.preview_view is None:
            return
        try:
            dlg.update_idletasks()
            px = self.preview_view.winfo_rootx()
            py = self.preview_view.winfo_rooty()
            pw = self.preview_view.winfo_width()
            ph = self.preview_view.winfo_height()
            dw = dlg.winfo_width()
            dh = dlg.winfo_height()
            x = px + max(0, (pw - dw) // 2)
            y = py + max(0, ph // 6)
            screen_h = dlg.winfo_screenheight()
            y = min(y, max(0, screen_h - dh - 40))
            dlg.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _after_correction_added(self, idx: int, old_label: str, new_label: str, confirmed: bool) -> None:
        if confirmed:
            self._log(f"[Validation manuelle] Noyau #{idx} : prédiction « {old_label} » confirmée correcte.\n")
        else:
            self._log(
                f"[Validation manuelle] Noyau #{idx} : « {old_label} » -> "
                f"« {new_label} » (corrigé manuellement).\n"
            )
        self._update_corrections_status()
        self._rerender_with_corrections()

    def _update_corrections_status(self) -> None:
        stats = corrections_summary(self.corrections)
        if stats["total"] == 0:
            self.corrections_status_var.set("0 noyau validé manuellement.")
        else:
            self.corrections_status_var.set(
                f"{stats['total']} noyau(x) validé(s) manuellement "
                f"({stats['confirmed']} confirmé(s), {stats['corrected']} corrigé(s))."
            )

    def _rerender_with_corrections(self) -> None:
        """Réapplique les corrections manuelles aux prédictions brutes déjà
        en mémoire et rafraîchit l'aperçu + le comptage instantanément.

        SANS relancer classify_and_render() : aucun rééchantillonnage radial
        n'est refait, seuls le dessin des cercles et les comptages sont
        reconstruits -> quasi instantané, même sur des dizaines de milliers
        de noyaux. C'est ce qui permet d'enlever/déplacer le cercle d'un
        noyau corrigé et de corriger le rapport sans attendre un recalcul
        complet des seuils.
        """
        if self.cached_data is None or self.raw_results is None:
            return
        try:
            cfg = self._collect_config()
        except ValueError:
            return
        composite = self.cached_data["composite"]
        t0 = time.time()
        try:
            result = rerender_with_corrections(composite, self.raw_results, self.corrections, cfg)
        except Exception:
            self._log(
                "\n--- ERREUR (application de la correction) ---\n"
                + traceback.format_exc() + "\n"
            )
            return
        self._apply_tuning_result(result)
        elapsed = time.time() - t0
        self._log(f"[Validation manuelle] Aperçu et rapport recalculés ({elapsed * 1000:.0f} ms).\n")

    # ── Sauvegarde / chargement / export des validations ─────────────────

    def _save_corrections_dialog(self) -> None:
        if not self.corrections:
            messagebox.showinfo("Aucune validation", "Aucun noyau validé pour le moment.")
            return
        out_dir = self.output_dir.get().strip() or str(Path.home())
        default_name = f"validations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = filedialog.asksaveasfilename(
            title="Sauver les validations manuelles", initialdir=out_dir,
            initialfile=default_name, defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        meta = {
            "czi_path": self.czi_path.get(),
            "n_nuclei": len(self.raw_results) if self.raw_results is not None else None,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            save_corrections(path, self.corrections, meta=meta)
        except Exception as e:
            messagebox.showerror("Erreur", f"Échec de la sauvegarde :\n{e}")
            return
        self._log(f"[Validation manuelle] {len(self.corrections)} validation(s) sauvée(s) -> {path}\n")
        messagebox.showinfo("Validations sauvées", f"Enregistrées dans :\n{path}")

    def _load_corrections_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Charger des validations manuelles", filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            loaded, meta = load_corrections(path)
        except Exception as e:
            messagebox.showerror("Erreur", f"Échec du chargement :\n{e}")
            return

        n_current = len(self.raw_results) if self.raw_results is not None else None
        n_saved = meta.get("n_nuclei")
        czi_saved = meta.get("czi_path")
        warn = ""
        if czi_saved and self.czi_path.get() and czi_saved != self.czi_path.get():
            warn += f"\n⚠ Fichier différent : sauvé pour « {czi_saved} »."
        if n_current is not None and n_saved is not None and n_current != n_saved:
            warn += (
                f"\n⚠ Nombre de noyaux différent : {n_saved} au moment de la "
                f"sauvegarde vs {n_current} actuellement — les index de "
                f"noyaux risquent de ne plus correspondre au bon endroit."
            )
        if warn:
            if not messagebox.askyesno(
                "Incohérence détectée",
                f"Ce fichier de validations ne semble pas correspondre à la "
                f"détection actuelle :{warn}\n\nCharger quand même ?",
            ):
                return

        self.corrections = loaded
        self._update_corrections_status()
        self._log(f"[Validation manuelle] {len(loaded)} validation(s) chargée(s) <- {path}\n")
        self._rerender_with_corrections()

    def _clear_corrections(self) -> None:
        if not self.corrections:
            return
        if not messagebox.askyesno(
            "Effacer les validations",
            f"Effacer les {len(self.corrections)} validation(s) manuelle(s) en cours ?",
        ):
            return
        self.corrections = {}
        self._update_corrections_status()
        self._log("[Validation manuelle] Toutes les validations ont été effacées.\n")
        self._rerender_with_corrections()

    def _on_export_csv_clicked(self) -> None:
        if self.cached_data is None or self.raw_results is None:
            messagebox.showwarning(
                "Aucune donnée",
                "Détectez et classifiez d'abord les noyaux (mode réglage "
                "interactif) avant d'exporter les features.",
            )
            return
        try:
            cfg = self._collect_config()
        except ValueError as e:
            messagebox.showerror("Paramètre invalide", str(e))
            return

        out_dir = self.output_dir.get().strip() or str(Path.home() / "cell_classifier_output")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        default_name = f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(
            title="Exporter les features (CSV)", initialdir=out_dir,
            initialfile=default_name, defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return

        self.export_csv_btn.configure(state="disabled")
        self.export_progress.start(12)
        self.export_status_var.set("⏳ Extraction des features en cours…")
        self._log(f"[Export CSV] Extraction des features pour {len(self.raw_results)} noyaux…\n")

        corrections_snapshot = dict(self.corrections)
        threading.Thread(
            target=self._export_worker,
            args=(cfg, self.cached_data, corrections_snapshot, path),
            daemon=True,
        ).start()

    def _export_worker(self, cfg: Config, cached: dict, corrections: dict, path: str) -> None:
        writer = QueueWriter(self.log_queue)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = writer
        sys.stderr = writer
        t0 = time.time()
        try:
            nuclei = cached["nuclei"]
            white_u8 = cached["white_u8"]
            green_u8 = cached["green_u8"]
            red_u8 = cached["red_u8"]
            blue_u8 = cached["blue_u8"]

            if self._last_bg_maps is not None and self._last_bg_globals is not None:
                bg_green, bg_red, bg_blue = self._last_bg_maps
                green_bg_global, red_bg_global, blue_bg_global = self._last_bg_globals
            else:
                bg_green = build_bg_map(green_u8, cfg.bg_grid)
                bg_red = build_bg_map(red_u8, cfg.bg_grid)
                bg_blue = build_bg_map(blue_u8, cfg.bg_grid)
                green_bg_global = float(np.median(green_u8))
                red_bg_global = float(np.median(red_u8))
                blue_bg_global = float(np.median(blue_u8))

            features, feature_names = extract_features(
                nuclei, green_u8, red_u8, blue_u8, white_u8,
                bg_green, bg_red, bg_blue,
                green_bg_global, red_bg_global, blue_bg_global,
                cfg,
            )
            export_features_csv(path, features, feature_names, self.raw_results, nuclei, corrections)
            elapsed = time.time() - t0
            self.tuning_queue.put(("export_ok", {"path": path, "elapsed": elapsed, "n": len(nuclei)}))
        except Exception:
            self.tuning_queue.put(("export_error", traceback.format_exc()))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def main() -> None:
    root = Tk()
    CellClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
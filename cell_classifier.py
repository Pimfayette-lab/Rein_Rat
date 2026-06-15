"""
cell_classifier.py
------------------
Step 1: Detect all nuclei in the white channel (from count_white_kernels v3).
Step 2: For each nucleus centroid, sample the colour channels in a search ring
        around the nucleus and classify the cell as:

  GREEN  – Very bright green signal in a ring AROUND the nucleus.
           After removing the nucleus the signal forms a solid block/donut.
           Signature: high green in the dilated ring, centred on nucleus.

  RED    – Red signal overlapping WITH the nucleus (and around it).
           Removing the nucleus does NOT leave a dedicated hole.
           Signature: high red in the full disk (nucleus + halo), bright
           relative to the red channel background median (mirrors GREEN logic).

  BLUE   – Blue signal that is off-centre relative to the nucleus.
           The nucleus is NOT the centre of the blue region.
           Signature: high blue in a half-ring / asymmetric halo; the
           centre-of-mass of the blue signal is displaced from the nucleus.

Classification priority: GREEN > RED > BLUE > UNCLASSIFIED
(a cell can only belong to one type; pick the strongest match)

Outputs
-------
  annotated_green.png   – composite with green cells marked
  annotated_red.png     – composite with red cells marked
  annotated_blue.png    – composite with blue cells marked
  annotated_all.png     – all three types on one image
  cell_report.txt       – text summary
"""

import numpy as np
import cv2
from czifile import imread
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG  – tune these to match your image
# ═══════════════════════════════════════════════════════════════════════════════
CZI_PATH = r"C:\Users\pimfa\Documents\MAIA\Rein_de_rat\transfer_12956536_files_7b6ebfa1\2021_05_18__0976gtAQP2nov_rbAE1.czi"

# ── Nucleus detection (same as v3) ───────────────────────────────────────────
MIN_AREA    = 50
MAX_AREA    = 80000
MERGE_RATIO = 1.6

# ── Search geometry (in pixels) ──────────────────────────────────────────────
# Inner radius  = nucleus boundary (signal inside this circle is "nucleus")
# Outer radius  = how far out we look for colour signal
NUCLEUS_R   = 12   # px  ← approximate nucleus radius; tune to your data
HALO_R      = 38   # px  ← outer edge of the search ring
# For GREEN: we check the ring between NUCLEUS_R and HALO_R
# For RED:   we check the disk of radius HALO_R (includes nucleus)
# For BLUE:  we check the ring AND measure centre-of-mass offset

# ── Classification thresholds (0-255 after uint8 normalisation) ──────────────
GREEN_THRESH        = 30    # mean green in ring to call it green
GREEN_BG_RATIO      = 4.0  # ring signal must be GREEN_BG_RATIO× the image median
# RED: mirrors GREEN logic but uses the full disk (nucleus included, no hole)
RED_THRESH          = 20    # mean red in full disk (radius HALO_R) to call it red
RED_BG_RATIO        = 3.0  # disk signal must be RED_BG_RATIO× red channel bg median
# Extra guard: red disk signal must also exceed green disk signal
# (avoids misclassifying green cells whose red channel has a small leak)
RED_NOT_GREENER     = False  # if True, red disk mean must be > green disk mean
BLUE_THRESH         = 20    # mean blue in ring to call it blue
BLUE_OFFSET_THRESH  = 0.3   # offset of blue CoM / HALO_R to call it off-centre

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def to_uint8(arr):
    arr = arr.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return (arr * 255).astype(np.uint8)


def make_ring_mask(shape, cx, cy, r_inner, r_outer):
    """Boolean mask: True for pixels in the ring r_inner < d <= r_outer."""
    H, W = shape
    ys = np.arange(max(0, cy - r_outer), min(H, cy + r_outer + 1))
    xs = np.arange(max(0, cx - r_outer), min(W, cx + r_outer + 1))
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    d2 = (yy - cy)**2 + (xx - cx)**2
    ring = (d2 > r_inner**2) & (d2 <= r_outer**2)
    # Return local coords and mask
    return ys, xs, ring


def sample_channel(ch, cx, cy, r_inner, r_outer):
    """
    Returns mean pixel value inside the ring [r_inner, r_outer],
    and the centre-of-mass displacement of bright pixels in the ring.
    """
    ys, xs, ring = make_ring_mask(ch.shape, cx, cy, r_inner, r_outer)
    patch = ch[np.ix_(ys, xs)]
    vals  = patch[ring].astype(np.float32)
    mean_val = float(vals.mean()) if vals.size else 0.0

    # Centre of mass of signal in ring (for BLUE offset test)
    if vals.sum() > 0:
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        w = patch[ring].astype(np.float64)
        com_y = float((yy[ring] * w).sum() / w.sum())
        com_x = float((xx[ring] * w).sum() / w.sum())
        offset = np.sqrt((com_x - cx)**2 + (com_y - cy)**2)
    else:
        offset = 0.0

    return mean_val, offset


def sample_disk(ch, cx, cy, r):
    """Mean inside a filled disk of radius r."""
    ys, xs, ring = make_ring_mask(ch.shape, cx, cy, 0, r)
    patch = ch[np.ix_(ys, xs)]
    vals  = patch[ring].astype(np.float32)
    return float(vals.mean()) if vals.size else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  1. LOAD IMAGE
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading CZI …")
img = np.squeeze(imread(CZI_PATH))
print(f"Shape: {img.shape}  dtype: {img.dtype}")

# Channels: 0=white, 1=green, 2=red, 3=blue  (adjust if yours differ)
white_raw = img[0]
green_raw = img[1]
red_raw   = img[2]
blue_raw  = img[3]

white_u8 = to_uint8(white_raw)
green_u8 = to_uint8(green_raw)
red_u8   = to_uint8(red_raw)
blue_u8  = to_uint8(blue_raw)

# Background medians (used to reject dim background signal, same logic for G and R)
green_bg_median = float(np.median(green_u8))
red_bg_median   = float(np.median(red_u8))
print(f"Green channel background median: {green_bg_median:.1f}")
print(f"Red   channel background median: {red_bg_median:.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  2. DETECT NUCLEI (identical to v3)
# ═══════════════════════════════════════════════════════════════════════════════
print("Detecting nuclei …")
blur   = cv2.GaussianBlur(white_u8, (5, 5), 1)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask   = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=1)

num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

blobs = []
for i in range(1, num_labels):
    area = int(stats[i, cv2.CC_STAT_AREA])
    if MIN_AREA <= area <= MAX_AREA:
        blobs.append((i, area, int(centroids[i][0]), int(centroids[i][1])))

areas = np.array([b[1] for b in blobs])
cutoff = np.percentile(areas, 70)
median_single = float(np.median(areas[areas <= cutoff]))
print(f"Blobs: {len(blobs)}  | median single-nucleus area: {median_single:.0f} px²")

# Build list of individual nucleus centroids (expanding merged blobs)
nuclei = []   # list of (cx, cy)
for (lid, area, cx, cy) in blobs:
    n = 1 if area < MERGE_RATIO * median_single else max(1, round(area / median_single))
    # For merged blobs we keep the same centroid (best we can do without watershed)
    for _ in range(n):
        nuclei.append((cx, cy))

print(f"Total nucleus count: {len(nuclei)}")

# ═══════════════════════════════════════════════════════════════════════════════
#  3. CLASSIFY EACH NUCLEUS
# ═══════════════════════════════════════════════════════════════════════════════
print("Classifying cells … (this may take a minute on 18k×18k)")

# Each channel is tested INDEPENDENTLY — no priority system.
# A nucleus can belong to multiple families simultaneously.
results = []   # list of (cx, cy, is_green, is_red, is_blue)

H, W = white_u8.shape

for idx, (cx, cy) in enumerate(nuclei):
    if idx % 5000 == 0:
        print(f"  … {idx}/{len(nuclei)}")

    # ── GREEN: bright ring centred on nucleus ─────────────────────────────────
    g_ring, _ = sample_channel(green_u8, cx, cy, NUCLEUS_R, HALO_R)
    is_green  = (g_ring >= GREEN_THRESH) and (g_ring >= GREEN_BG_RATIO * green_bg_median)

    # ── RED: full disk, mirrors GREEN logic ───────────────────────────────────
    r_disk = sample_disk(red_u8,   cx, cy, HALO_R)
    g_disk = sample_disk(green_u8, cx, cy, HALO_R)
    is_red = (r_disk >= RED_THRESH) and (r_disk >= RED_BG_RATIO * red_bg_median)
    if RED_NOT_GREENER and is_red:
        is_red = r_disk > g_disk

    # ── BLUE: off-centre signal ───────────────────────────────────────────────
    b_ring, b_offset = sample_channel(blue_u8, cx, cy, NUCLEUS_R, HALO_R)
    is_blue = (b_ring >= BLUE_THRESH) and (b_offset >= BLUE_OFFSET_THRESH * HALO_R)

    results.append((cx, cy, is_green, is_red, is_blue))

# ── Per-family lists (a nucleus can appear in multiple) ───────────────────────
green_cells  = [(cx, cy) for cx, cy, g, r, b in results if g]
red_cells    = [(cx, cy) for cx, cy, g, r, b in results if r]
blue_cells   = [(cx, cy) for cx, cy, g, r, b in results if b]
unclassified = [(cx, cy) for cx, cy, g, r, b in results if not g and not r and not b]

# ── Co-expression breakdown ───────────────────────────────────────────────────
only_green = [(cx, cy) for cx, cy, g, r, b in results if     g and not r and not b]
only_red   = [(cx, cy) for cx, cy, g, r, b in results if not g and     r and not b]
only_blue  = [(cx, cy) for cx, cy, g, r, b in results if not g and not r and     b]
green_red  = [(cx, cy) for cx, cy, g, r, b in results if g and r and not b]
green_blue = [(cx, cy) for cx, cy, g, r, b in results if g and not r and b]
red_blue   = [(cx, cy) for cx, cy, g, r, b in results if not g and r and b]
all_three  = [(cx, cy) for cx, cy, g, r, b in results if g and r and b]

print(f"GREEN: {len(green_cells)}  RED: {len(red_cells)}  BLUE: {len(blue_cells)}  "
      f"UNCLASSIFIED: {len(unclassified)}")
print(f"Co-expr — G+R: {len(green_red)}  G+B: {len(green_blue)}  "
      f"R+B: {len(red_blue)}  G+R+B: {len(all_three)}")

# ═══════════════════════════════════════════════════════════════════════════════
#  4. BUILD COMPOSITE VISUALISATION BASE
# ═══════════════════════════════════════════════════════════════════════════════
# Merge all 4 channels into a false-colour RGB for background
composite = np.zeros((H, W, 3), dtype=np.uint8)
composite[:, :, 1] = (green_u8 * 0.6).astype(np.uint8)   # G → green
composite[:, :, 2] = (red_u8   * 0.6).astype(np.uint8)   # R → red
composite[:, :, 0] = (blue_u8  * 0.6).astype(np.uint8)   # B → blue (OpenCV BGR)
# White channel blended in as brightness
w_contrib = (white_u8 * 0.4).astype(np.uint8)
composite = np.clip(composite.astype(np.int16) + w_contrib[:, :, None], 0, 255).astype(np.uint8)

def draw_circles(img, cells, colour, radius=8, thickness=2):
    for (cx, cy) in cells:
        cv2.circle(img, (cx, cy), radius, colour, thickness)

# ── Individual channel images ──────────────────────────────────────────────────
vis_green = composite.copy(); draw_circles(vis_green, green_cells, (0, 255, 0))
vis_red   = composite.copy(); draw_circles(vis_red,   red_cells,   (0, 0, 255))
vis_blue  = composite.copy(); draw_circles(vis_blue,  blue_cells,  (255, 80, 0))

cv2.imwrite("annotated_green.png", vis_green)
cv2.imwrite("annotated_red.png",   vis_red)
cv2.imwrite("annotated_blue.png",  vis_blue)

# ── Combined image: colour encodes which families a nucleus belongs to ─────────
#   Single:      green / red / blue circle
#   Co-expressed: mixed colour + outer ring to stand out
#   G+R = yellow,  G+B = cyan,  R+B = magenta,  G+R+B = white
vis_all = composite.copy()

# Draw single-positive first (smallest radius), co-expressed on top (larger)
draw_circles(vis_all, only_green, (0, 255, 0),     radius=7)
draw_circles(vis_all, only_red,   (0, 0, 255),     radius=7)
draw_circles(vis_all, only_blue,  (255, 80, 0),    radius=7)
draw_circles(vis_all, green_red,  (0, 255, 255),   radius=10)   # yellow (BGR)
draw_circles(vis_all, green_blue, (255, 255, 0),   radius=10)   # cyan
draw_circles(vis_all, red_blue,   (255, 0, 255),   radius=10)   # magenta
draw_circles(vis_all, all_three,  (255, 255, 255), radius=13)   # white

cv2.imwrite("annotated_all.png", vis_all)
print("Annotated images saved.")

# ═══════════════════════════════════════════════════════════════════════════════
#  5. TEXT REPORT
# ═══════════════════════════════════════════════════════════════════════════════
total = len(nuclei)

def pct(n):
    return f"{100*n/total:.1f}%" if total > 0 else "N/A"

report = f"""
╔══════════════════════════════════════════════════════╗
║           CELL CLASSIFICATION REPORT                 ║
╚══════════════════════════════════════════════════════╝
Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Source    : {CZI_PATH}

── NUCLEUS DETECTION ────────────────────────────────────
  Connected-component blobs (area {MIN_AREA}–{MAX_AREA} px²) : {len(blobs)}
  Estimated single-nucleus area                       : {median_single:.0f} px²
  TOTAL NUCLEI (after merged-blob split)              : {total}

── CELL TYPE COUNTS (multi-label: one nucleus can appear in several) ────
  GREEN  (any)                           : {len(green_cells):6d}  ({pct(len(green_cells))})
  RED    (any)                           : {len(red_cells):6d}  ({pct(len(red_cells))})
  BLUE   (any)                           : {len(blue_cells):6d}  ({pct(len(blue_cells))})
  UNCLASSIFIED                           : {len(unclassified):6d}  ({pct(len(unclassified))})

── CO-EXPRESSION BREAKDOWN ──────────────────────────────
  Green only                             : {len(only_green):6d}  ({pct(len(only_green))})
  Red   only                             : {len(only_red):6d}  ({pct(len(only_red))})
  Blue  only                             : {len(only_blue):6d}  ({pct(len(only_blue))})
  Green + Red                            : {len(green_red):6d}  ({pct(len(green_red))})
  Green + Blue                           : {len(green_blue):6d}  ({pct(len(green_blue))})
  Red   + Blue                           : {len(red_blue):6d}  ({pct(len(red_blue))})
  Green + Red + Blue                     : {len(all_three):6d}  ({pct(len(all_three))})
  ─────────────────────────────────────────────────────
  TOTAL nuclei                           : {total:6d}

── DETECTION PARAMETERS ────────────────────────────────
  Nucleus radius  (NUCLEUS_R)  : {NUCLEUS_R} px
  Halo radius     (HALO_R)     : {HALO_R} px
  Green ring threshold         : {GREEN_THRESH}  (bg median: {green_bg_median:.1f})
  Green bg ratio               : {GREEN_BG_RATIO}×
  Red disk threshold           : {RED_THRESH}  (bg median: {red_bg_median:.1f})
  Red bg ratio                 : {RED_BG_RATIO}×
  Red not-greener guard        : {RED_NOT_GREENER}
  Blue ring threshold          : {BLUE_THRESH}
  Blue offset threshold        : {BLUE_OFFSET_THRESH}× HALO_R = {BLUE_OFFSET_THRESH*HALO_R:.1f} px

── OUTPUT FILES ─────────────────────────────────────────
  annotated_green.png  – green cells highlighted
  annotated_red.png    – red cells highlighted
  annotated_blue.png   – blue cells highlighted
  annotated_all.png    – all three types combined
  cell_report.txt      – this report
"""

print(report)
with open("cell_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("Report saved → cell_report.txt")
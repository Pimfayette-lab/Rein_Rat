"""
count_white_kernels.py  (v3 – CC base + area-based split for merged blobs)
--------------------------------------------------------------------------
Strategy:
  1. Connected-components (fast, works at 18k×18k scale)
  2. For each blob:
       - if area < 1.5× median nucleus area  → count as 1
       - else estimate n = round(area / median_area) → count as n
  This avoids the broken peak_local_max approach on large images.
"""

import numpy as np
import cv2
from czifile import imread

# ── CONFIG ────────────────────────────────────────────────────────────────────
CZI_PATH = r"C:\Users\pimfa\Documents\MAIA\Rein_de_rat\transfer_12956536_files_7b6ebfa1\2021_05_18__0973gtAQP2nov_rbAE1.czi"

MIN_AREA        = 50     # px²  — drop speckles smaller than this
MAX_AREA        = 80000  # px²  — drop giant artefacts (staining blobs etc.)
# A blob larger than MERGE_RATIO × median_single_nucleus_area is
# considered a cluster of merged nuclei and its count = round(area / median)
MERGE_RATIO     = 1.6

# ── 1. Load & extract white channel ──────────────────────────────────────────
img = np.squeeze(imread(CZI_PATH))
print(f"Full image shape : {img.shape}  dtype: {img.dtype}")

white_raw = img[0]
print(f"White channel    : {white_raw.shape}  min={white_raw.min()}  max={white_raw.max()}")

# ── 2. Normalise → uint8 ─────────────────────────────────────────────────────
def to_uint8(arr):
    arr = arr.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return (arr * 255).astype(np.uint8)

white_u8 = to_uint8(white_raw)

# ── 3. Blur + Otsu threshold ─────────────────────────────────────────────────
blur = cv2.GaussianBlur(white_u8, (5, 5), 1)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Opening: remove tiny speckles
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=1)

cv2.imwrite("white_channel_mask.png", mask)
print("Binary mask saved → white_channel_mask.png")

# ── 4. Connected components ───────────────────────────────────────────────────
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    mask, connectivity=8
)
print(f"Raw CC (excl. background): {num_labels - 1}")

# ── 5. Collect blobs in valid area range ─────────────────────────────────────
blobs = []   # (label_id, area, cx, cy)
for i in range(1, num_labels):
    area = int(stats[i, cv2.CC_STAT_AREA])
    if MIN_AREA <= area <= MAX_AREA:
        cx = int(centroids[i][0])
        cy = int(centroids[i][1])
        blobs.append((i, area, cx, cy))

areas = np.array([b[1] for b in blobs])
print(f"Blobs after area filter [{MIN_AREA}–{MAX_AREA} px²]: {len(blobs)}")

# ── 6. Estimate single-nucleus area from the distribution ────────────────────
# The mode / lower half of the area histogram represents single nuclei.
# We take the median of the smallest 70% of blobs as our reference size.
cutoff = np.percentile(areas, 70)
single_areas = areas[areas <= cutoff]
median_single = float(np.median(single_areas)) if len(single_areas) else float(np.median(areas))
print(f"Estimated single-nucleus area: {median_single:.1f} px²")

# ── 7. Count nuclei (1 per blob, or estimated n if blob is a cluster) ────────
total_nuclei = 0
annotations  = []   # (cx, cy, count_in_blob)

for (lid, area, cx, cy) in blobs:
    if area < MERGE_RATIO * median_single:
        n = 1
    else:
        n = max(1, round(area / median_single))
    total_nuclei += n
    annotations.append((cx, cy, n))

print(f"\n{'='*54}")
print(f"  Valid blobs (connected components) : {len(blobs)}")
print(f"  Estimated single-nucleus area      : {median_single:.0f} px²")
print(f"  TOTAL WHITE NUCLEI (with splits)   : {total_nuclei}")
print(f"{'='*54}\n")

# ── 8. Annotated output ───────────────────────────────────────────────────────
vis = cv2.cvtColor(white_u8, cv2.COLOR_GRAY2BGR)

for (cx, cy, n) in annotations:
    color = (0, 0, 255) if n == 1 else (0, 165, 255)   # red=single, orange=cluster
    cv2.circle(vis, (cx, cy), 5 if n == 1 else 8, color, 2)
    if n > 1:
        cv2.putText(vis, f"x{n}", (cx + 6, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

cv2.imwrite("white_channel_counted.png", vis)
print("Annotated image saved → white_channel_counted.png")
print(f">>> TOTAL WHITE NUCLEI: {total_nuclei} <<<")
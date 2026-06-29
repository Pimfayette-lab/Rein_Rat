import numpy as np
import cv2
from czifile import imread
from scipy import ndimage as ndi

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CZI_PATH = r"C:\Users\pimfa\Documents\MAIA\Rein_de_rat\transfer_12956536_files_7b6ebfa1\2021_05_18__0986gtAQP2nov_rbAE1.czi"

MIN_AREA = 50
MAX_AREA = 80000

MERGE_RATIO = 1.6
CIRCULARITY_TH = 0.65
SOLIDITY_TH = 0.90

# ─────────────────────────────────────────────────────────────
# 1. LOAD IMAGE
# ─────────────────────────────────────────────────────────────
img = np.squeeze(imread(CZI_PATH))
white_raw = img[0]

def to_uint8(arr):
    arr = arr.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return (arr * 255).astype(np.uint8)

white = to_uint8(white_raw)

# ─────────────────────────────────────────────────────────────
# 2. PREPROCESS + THRESHOLD
# ─────────────────────────────────────────────────────────────
blur = cv2.GaussianBlur(white, (5, 5), 1)
_, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

# ─────────────────────────────────────────────────────────────
# 3. CONNECTED COMPONENTS
# ─────────────────────────────────────────────────────────────
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

# ─────────────────────────────────────────────────────────────
# 4. COLLECT BLOBS + SHAPE FEATURES
# ─────────────────────────────────────────────────────────────
blobs = []

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area < MIN_AREA or area > MAX_AREA:
        continue

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    blob_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8)

    # perimeter
    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        continue

    perimeter = cv2.arcLength(contours[0], True) + 1e-6
    circularity = 4 * np.pi * area / (perimeter ** 2)

    hull = cv2.convexHull(contours[0])
    hull_area = cv2.contourArea(hull) + 1e-6
    solidity = area / hull_area

    cx, cy = centroids[i]

    blobs.append({
        "id": i,
        "area": area,
        "cx": int(cx),
        "cy": int(cy),
        "bbox": (x, y, w, h),
        "circularity": circularity,
        "solidity": solidity
    })

areas = np.array([b["area"] for b in blobs])
median_single = np.median(areas[areas < np.percentile(areas, 70)])

print("Estimated single nucleus area:", median_single)

# ─────────────────────────────────────────────────────────────
# 5. WATERSHED SPLITTER (for clusters)
# ─────────────────────────────────────────────────────────────
def split(blob):
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)

    # adaptive threshold (key upgrade)
    peaks = (dist > (0.5 * dist.max() + 0.5 * dist.mean())).astype(np.uint8)
    
    return max(1, cv2.connectedComponents(peaks)[0] - 1)

# ─────────────────────────────────────────────────────────────
# 6. COUNT NUCLEI
# ─────────────────────────────────────────────────────────────
total = 0
annotations = []

for b in blobs:
    i = b["id"]
    area = b["area"]

    x, y, w, h = b["bbox"]
    full_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8)

    is_large = area > MERGE_RATIO * median_single
    is_irregular = (b["circularity"] < CIRCULARITY_TH) or (b["solidity"] < SOLIDITY_TH)

    if not (is_large or is_irregular):
        n = 1
    else:
        n = split(full_mask)

    total += n
    annotations.append((b["cx"], b["cy"], n))

# ─────────────────────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────────────────────
vis = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)

for cx, cy, n in annotations:
    color = (0, 0, 255) if n == 1 else (0, 165, 255)
    cv2.circle(vis, (cx, cy), 6, color, 2)
    if n > 1:
        cv2.putText(vis, f"x{n}", (cx+5, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

cv2.imwrite("improved_nuclei_count.png", vis)

print("\n==============================")
print("TOTAL NUCLEI:", total)
print("==============================")
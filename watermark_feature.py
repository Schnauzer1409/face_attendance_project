import cv2
import numpy as np
import pickle
from pathlib import Path
from typing import Optional

# =====================================================
# CẤU HÌNH
# =====================================================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

WM_FEATURE_FILE = DATA_DIR / "watermark_feature.pkl"

# ORB detector (nhẹ – ổn định)
ORB = cv2.ORB.create(
    nfeatures=2000,
    scaleFactor=1.2,
    nlevels=8
)

# =====================================================
# TRÍCH XUẤT ĐẶC TRƯNG TỪ 1 ẢNH
# =====================================================
def extract_feature(image_bgr) -> Optional[np.ndarray]:
    """
    image_bgr: ảnh OpenCV (BGR)
    return: descriptors (numpy array) hoặc None
    """

    if image_bgr is None:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ⚠️ Pylance không hiểu C-extension → ignore type
    keypoints, descriptors = ORB.detectAndCompute(gray, None)  # type: ignore

    if descriptors is None or len(descriptors) < 20:
        return None

    return descriptors


# =====================================================
# TRAIN WATERMARK (NHIỀU ẢNH → 1 FEATURE)
# =====================================================
def train_watermark(image_list):
    """
    image_list: list ảnh OpenCV (BGR)
    """

    all_desc = []

    for img in image_list:
        desc = extract_feature(img)
        if desc is not None:
            all_desc.append(desc)

    if len(all_desc) < 3:
        raise ValueError("Không đủ đặc trưng watermark để train")

    # Gộp tất cả descriptor
    all_desc = np.vstack(all_desc)

    with open(WM_FEATURE_FILE, "wb") as f:
        pickle.dump(all_desc, f)

    return {
        "status": "ok",
        "total_descriptors": int(len(all_desc))
    }


# =====================================================
# VERIFY WATERMARK (SO ẢNH MỚI VỚI FEATURE ĐÃ TRAIN)
# =====================================================
def verify_watermark(image_bgr, min_matches: int = 50):
    """
    image_bgr: ảnh cần kiểm tra
    min_matches: ngưỡng khớp
    """

    if not WM_FEATURE_FILE.exists():
        return False, 0

    desc_query = extract_feature(image_bgr)
    if desc_query is None:
        return False, 0

    with open(WM_FEATURE_FILE, "rb") as f:
        desc_train = pickle.load(f)

    # BFMatcher cho ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(desc_query, desc_train)

    # Sắp xếp theo độ tốt
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = [m for m in matches if m.distance < 50]

    score = len(good_matches)

    is_valid = score >= min_matches

    return is_valid, score

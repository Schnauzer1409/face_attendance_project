# ===============================
# IMPORT THƯ VIỆN
# ===============================
from watermark_feature import train_watermark, verify_watermark
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import sqlite3
import base64
import io
import time
import pickle

import numpy as np
from PIL import Image

from insightface.app import FaceAnalysis


# ===============================
# KHAI BÁO THƯ MỤC
# ===============================
BASE_DIR = Path(__file__).resolve().parent

SAVE_DIR = BASE_DIR / "uploads"
ENROLL_DIR = SAVE_DIR / "enroll"        # ảnh đăng ký khuôn mặt
ATT_DIR = SAVE_DIR / "attendance"       # ảnh điểm danh
WM_DIR = BASE_DIR / "watermarks"        # ảnh watermark phòng

SAVE_DIR.mkdir(exist_ok=True)
ENROLL_DIR.mkdir(exist_ok=True)
ATT_DIR.mkdir(exist_ok=True)
WM_DIR.mkdir(exist_ok=True)


# ===============================
# KHỞI TẠO FASTAPI
# ===============================
app = FastAPI()

FRONTEND_DIR = BASE_DIR.parent / "Frontend"

# Cho phép load HTML / CSS / JS
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ===============================
# ROUTE GIAO DIỆN
# ===============================
@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/teacher")
def teacher_page():
    return FileResponse(FRONTEND_DIR / "teacher.html")

@app.get("/student")
def student_page():
    return FileResponse(FRONTEND_DIR / "student.html")


# ===============================
# DATABASE SQLITE
# ===============================
DB_PATH = BASE_DIR / "attendance.db"

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def sql(query, params=()):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(query, params)
    conn.commit()
    rows = cur.fetchall()
    conn.close()
    return rows


# Bảng user
sql("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT,
    token TEXT
)
""")

# Bảng lưu embedding khuôn mặt
sql("""
CREATE TABLE IF NOT EXISTS encodings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    embedding BLOB
)
""")


# ===============================
# TẠO USER MẪU
# ===============================
def add_default_users():
    users = sql("SELECT username FROM users")
    existed = {u[0] for u in users}

    if "teacher1" not in existed:
        sql(
            "INSERT INTO users(username,password,role) VALUES (?,?,?)",
            ("teacher1", "123456", "teacher")
        )

    if "student1" not in existed:
        sql(
            "INSERT INTO users(username,password,role) VALUES (?,?,?)",
            ("student1", "123456", "student")
        )

add_default_users()


# ===============================
# LOAD MODEL NHẬN DIỆN
# ===============================
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))


# ===============================
# HÀM TIỆN ÍCH
# ===============================
# Chuyển base64 → numpy image
def decode_base64_to_image(b64):
    header, data = b64.split(",", 1)
    img_bytes = base64.b64decode(data)
    return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

# Tính độ giống cosine
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
import cv2

def check_watermark(att_img: np.ndarray, wm_img: np.ndarray, threshold=0.15):
    """
    att_img : ảnh điểm danh (RGB)
    wm_img  : ảnh watermark (RGB)
    """

    gray1 = cv2.cvtColor(att_img, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(wm_img, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB.create(nfeatures=500)

    mask1 = np.ones(gray1.shape, dtype=np.uint8)
    mask2 = np.ones(gray2.shape, dtype=np.uint8)

    kp1, des1 = orb.detectAndCompute(gray1, mask1)
    kp2, des2 = orb.detectAndCompute(gray2, mask2)

    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    score = len(matches) / max(len(kp2), 1)

    return score > threshold



# ===============================
# API LOGIN
# ===============================
@app.post("/api/login")
def login(username: str = Form(...), password: str = Form(...)):
    user = sql(
        "SELECT username,password,role FROM users WHERE username=?",
        (username,)
    )

    if not user:
        return {"status": "fail", "msg": "User không tồn tại"}

    db_user, db_pass, role = user[0]

    if password != db_pass:
        return {"status": "fail", "msg": "Sai mật khẩu"}

    token = f"{username}_{int(time.time())}"
    sql("UPDATE users SET token=? WHERE username=?", (token, username))

    return {
        "status": "ok",
        "username": username,
        "role": role,
        "token": token
    }


# ===============================
# ENROLL KHUÔN MẶT (GIÁO VIÊN)
# ===============================
@app.post("/api/enroll")
def enroll(username: str = Form(...), file: UploadFile = File(...)):
    img_bytes = file.file.read()

    # Lưu ảnh
    filename = f"{username}_{int(time.time())}.jpg"
    with open(ENROLL_DIR / filename, "wb") as f:
        f.write(img_bytes)

    # Nhận diện mặt
    img_np = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    faces = face_app.get(img_np)

    if len(faces) == 0:
        return {"status": "no_face"}

    # Lấy embedding
    emb = faces[0].embedding
    emb_blob = pickle.dumps(emb)

    # Lưu vào DB
    sql(
        "INSERT INTO encodings(username,embedding) VALUES (?,?)",
        (username, emb_blob)
    )

    return {"status": "ok", "msg": "Enroll thành công"}


# ===============================
# NHẬN DIỆN REALTIME
# ===============================
@app.post("/api/frame")
def frame(img: str = Form(...)):
    img_np = decode_base64_to_image(img)
    faces = face_app.get(img_np)

    if len(faces) == 0:
        return {"status": "noface"}

    emb = faces[0].embedding
    encs = sql("SELECT username, embedding FROM encodings")

    best_user = None
    best_score = -1

    for uname, blob in encs:
        known_emb = pickle.loads(blob)
        score = cosine(emb, known_emb)

        if score > best_score:
            best_score = score
            best_user = uname

    if best_score < 0.5:
        return {"status": "unknown", "score": best_score}

    return {
        "status": "ok",
        "recognized": best_user,
        "score": best_score
    }


# ===============================
# API ĐIỂM DANH
# ===============================
@app.post("/api/attendance")
def attendance(username: str = Form(...), file: UploadFile = File(...)):
    # ======================
    # 1. Đọc ảnh gửi lên
    # ======================
    img_bytes = file.file.read()

    filename = f"{username}_{int(time.time())}.jpg"
    with open(ATT_DIR / filename, "wb") as f:
        f.write(img_bytes)

    img_np = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

    # ======================
    # 2. CHECK KHUÔN MẶT
    # ======================
    faces = face_app.get(img_np)

    if len(faces) == 0:
        return {"status": "fail", "msg": "Không phát hiện khuôn mặt"}

    emb = faces[0].embedding
    encs = sql("SELECT username, embedding FROM encodings")

    best_user = None
    best_score = -1

    for uname, blob in encs:
        known_emb = pickle.loads(blob)
        score = cosine(emb, known_emb)

        if score > best_score:
            best_score = score
            best_user = uname

    if best_score < 0.6:
        return {"status": "fail", "msg": "Khuôn mặt không khớp"}

    if best_user != username:
        return {"status": "fail", "msg": "Sai người điểm danh"}

    # ======================
    # 3. CHECK WATERMARK (FEATURE MATCHING)
    # ======================
    is_valid, wm_score = verify_watermark(img_np)

    if not is_valid:
        return {
            "status": "fail",
            "msg": f"Sai watermark phòng học (score={wm_score})"
        }

    # ======================
    # 4. THÀNH CÔNG
    # ======================
    return {
        "status": "success",
        "msg": f"Điểm danh thành công cho {username}"
    }

# ===============================
# XÓA TOÀN BỘ ENCODING
# ===============================
@app.get("/api/clear_encodings")
def clear_encodings():
    sql("DELETE FROM encodings")
    return {"status": "ok"}


# ===============================
# WATERMARK (ẢNH PHÒNG)
# ===============================
@app.post("/api/teacher_generate_watermark")
def generate_watermark(file: UploadFile = File(...)):
    import cv2

    img_bytes = file.file.read()
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)

    h, w = gray.shape
    wm_size = 120
    best_score = -1
    bx = by = 0

    for y in range(0, h - wm_size, 20):
        for x in range(0, w - wm_size, 20):
            score = edges[y:y+wm_size, x:x+wm_size].sum()
            if score > best_score:
                best_score = score
                bx, by = x, y

    crop = img[by:by+wm_size, bx:bx+wm_size]

    # Lưu watermark tạm
    Image.fromarray(crop).save(WM_DIR / "temp_watermark.jpg")

    # Trả về base64 để frontend hiển thị
    buf = io.BytesIO()
    Image.fromarray(crop).save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "status": "ok",
        "watermark": b64
    }
# ===============================
# TRAIN WATERMARK (GIÁO VIÊN)
# ===============================
@app.post("/api/train_watermark")
def train_watermark_api(files: list[UploadFile] = File(...)):
    images = []

    for file in files:
        img_bytes = file.file.read()
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        images.append(img)

    try:
        result = train_watermark(images)
        return result
    except Exception as e:
        return {
            "status": "error",
            "msg": str(e)
        }

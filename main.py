from fastapi import FastAPI, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sqlite3
import secrets
import os
import cv2
from insightface.app import FaceAnalysis
import numpy as np


app = FastAPI()

# ============================
# STATIC FILES (Frontend)
# ============================
# Trỏ tới thư mục Frontend
app.mount("/static", StaticFiles(directory="Frontend"), name="static")


# ============================
# CORS
# ============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================
# DATABASE INIT
# ============================
def init_db():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()

    # user data
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            role TEXT,
            face BLOB
        )
    """)

    # attendance logs
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ============================
# FACE RECOGNITION
# ============================
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))


def extract_face_embedding(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    faces = face_app.get(img)

    if len(faces) == 0:
        return None

    return faces[0].embedding.tobytes()


def compare_face(emb1, emb2):
    e1 = np.frombuffer(emb1, dtype=np.float32)
    e2 = np.frombuffer(emb2, dtype=np.float32)
    similarity = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    return similarity > 0.5


# ============================
# LOGIN
# ============================
@app.post("/api/login")
async def login(username: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()

    c.execute("SELECT password, role FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()

    if row is None or row[0] != password:
        return JSONResponse({"status": "error", "msg": "Sai tài khoản hoặc mật khẩu"})

    token = secrets.token_hex(16)

    return JSONResponse({
        "status": "ok",
        "token": token,
        "username": username,
        "role": row[1]
    })


# ============================
# TEACHER: ENROLL
# ============================
@app.post("/api/enroll")
async def enroll(
    file: UploadFile,
    token: str = Form(...)
):
    img_bytes = await file.read()
    embedding = extract_face_embedding(img_bytes)

    if embedding is None:
        return JSONResponse({"status": "error", "msg": "Không thấy mặt trong ảnh"})

    # Trong bản demo: teacher chỉ enroll của chính mình
    username = "teacher1"

    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()

    c.execute("UPDATE users SET face=? WHERE username=?", (embedding, username))
    conn.commit()
    conn.close()

    return JSONResponse({"status": "ok", "msg": "Đã lưu khuôn mặt!"})


# ============================
# STUDENT: ATTENDANCE
# ============================
@app.post("/api/frame")
async def frame(
    file: UploadFile,
    token: str = Form(...)
):
    img_bytes = await file.read()
    embedding = extract_face_embedding(img_bytes)

    if embedding is None:
        return JSONResponse({"status": "error", "msg": "Không tìm thấy mặt"})

    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()

    c.execute("SELECT username, face FROM users WHERE face IS NOT NULL")
    rows = c.fetchall()

    for username, face_data in rows:
        if face_data and compare_face(embedding, face_data):
            c.execute("INSERT INTO logs (username) VALUES (?)", (username,))
            conn.commit()
            conn.close()
            return JSONResponse({"status": "ok", "msg": f"Điểm danh thành công: {username}"})

    conn.close()
    return JSONResponse({"status": "error", "msg": "Không khớp khuôn mặt nào"})


# ============================
# ROOT PAGE -> redirect to login
# ============================
@app.get("/")
async def root():
    return JSONResponse({"msg": "Server running. Open /static/index.html"})
    
document.addEventListener("DOMContentLoaded", () => {
    const path = location.pathname;
    const role = localStorage.getItem("role");

    // Redirect nếu đã login
    if (role === "teacher" && path.includes("index.html")) {
        window.location.href = "/static/teacher.html";
    }

    if (role === "student" && path.includes("index.html")) {
        window.location.href = "/static/student.html";
    }

    // ==== CAMERA INIT ====
    const video = document.getElementById("video");

    if (video) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
                window._cameraReady = true;   // camera đã mở
            })
            .catch(err => {
                console.error("Camera error:", err);
                document.getElementById("msg").innerText =
                    "Không thể mở camera! Hãy kiểm tra quyền truy cập.";
            });
    }
});

// ========================= LOGIN =========================
async function login() {
    let f = new FormData();
    f.append("username", document.getElementById("username").value);
    f.append("password", document.getElementById("password").value);

    let res = await fetch("/api/login", {
        method: "POST",
        body: f
    });

    let data = await res.json();

    if (data.status !== "ok") {
        document.getElementById("error").innerText = data.msg;
        return;
    }

    localStorage.setItem("token", data.token);
    localStorage.setItem("username", data.username);
    localStorage.setItem("role", data.role);

    if (data.role === "teacher")
        window.location.href = "/static/teacher.html";
    else
        window.location.href = "/static/student.html";
}

// ========================= LOGOUT =========================
function logout() {
    localStorage.clear();
    window.location.href = "/static/index.html";
}

// ========================= CAPTURE ATTENDANCE =========================
async function capture() {
    if (!window._cameraReady) {
        document.getElementById("msg").innerText = "Camera chưa sẵn sàng...";
        return;
    }

    const video = document.getElementById("video");
    const canvas = document.createElement("canvas");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const blob = await new Promise(resolve => canvas.toBlob(resolve));

    let f = new FormData();
    f.append("file", blob);
    f.append("token", localStorage.getItem("token"));

    let res = await fetch("/api/frame", {
        method: "POST",
        body: f
    });

    let data = await res.json();
    document.getElementById("msg").innerText = data.msg;
}

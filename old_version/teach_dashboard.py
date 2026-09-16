from flask import Flask, Response, render_template_string, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import threading
import os
import time
import shutil

app = Flask(__name__)

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "/home/vansh/smart-robo-arm/combined_robot_v3/weights/best.pt"

TEACH_DIR = "/home/vansh/smart-robo-arm/robot_teach_dataset"
PENDING_DIR = os.path.join(TEACH_DIR, "pending")
IMAGE_DIR = os.path.join(TEACH_DIR, "images")
LABEL_DIR = os.path.join(TEACH_DIR, "labels")

os.makedirs(PENDING_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# ============================================================
# YOLO
# ============================================================

model = YOLO(MODEL_PATH)

# Classes from your trained model
CLASS_NAMES = {
    0: "eraser",
    1: "glue",
    2: "marker",
    3: "pen",
    4: "pencil",
    5: "ruler",
    6: "scissor",
    7: "sharpener",
    8: "stapler",
    9: "tape"
}

# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")

latest_frame = None
frame_lock = threading.Lock()


def camera_loop():

    global latest_frame

    while True:

        success, frame = camera.read()

        if not success:
            continue

        with frame_lock:
            latest_frame = frame.copy()

        time.sleep(0.001)


camera_thread = threading.Thread(
    target=camera_loop,
    daemon=True
)

camera_thread.start()

# ============================================================
# LIVE YOLO STREAM
# ============================================================

def generate_frames():

    while True:

        with frame_lock:

            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        # YOLO
        results = model(
            frame,
            imgsz=416,
            conf=0.20,
            verbose=False
        )

        result = results[0]

        # Draw detections
        annotated = result.plot()

        # JPEG
        success, buffer = cv2.imencode(
            ".jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def index():

    html = """
<!DOCTYPE html>

<html>

<head>

<title>Robotic Arm Vision</title>

<style>

body {
    background: #111;
    color: white;
    font-family: Arial, sans-serif;
    margin: 0;
}

header {
    padding: 20px;
    text-align: center;
    background: #181818;
}

h1 {
    margin: 0;
}

.container {
    width: 1100px;
    max-width: 95%;
    margin: 20px auto;
}

.camera {
    text-align: center;
}

.camera img {
    width: 80%;
    max-width: 900px;
    border: 3px solid #444;
    border-radius: 10px;
}

.controls {
    margin-top: 20px;
    text-align: center;
}

button {
    padding: 12px 25px;
    font-size: 16px;
    border: none;
    border-radius: 7px;
    cursor: pointer;
}

#teachButton {
    background: #6c2cff;
    color: white;
}

#saveButton {
    background: #16a34a;
    color: white;
    margin-top: 15px;
}

#cancelButton {
    background: #444;
    color: white;
    margin-top: 15px;
}

#teachPanel {
    display: none;
    margin-top: 25px;
    padding: 20px;
    background: #1c1c1c;
    border-radius: 10px;
}

#canvasContainer {
    text-align: center;
}

canvas {
    border: 2px solid white;
    cursor: crosshair;
    max-width: 100%;
}

select {
    padding: 10px;
    font-size: 16px;
    margin-top: 10px;
}

.status {
    margin-top: 15px;
    font-size: 16px;
}

.instructions {
    color: #aaa;
    margin: 10px;
}

</style>

</head>

<body>

<header>

<h1>YOLO Object Detection</h1>

</header>

<div class="container">

<div class="camera">

<img src="/video_feed">

</div>

<div class="controls">

<button id="teachButton" onclick="startTeaching()">
Teach Object
</button>

</div>

<div id="teachPanel">

<h2>Teach Mode</h2>

<p class="instructions">
Draw a bounding box around the object you want the robot to learn.
</p>

<div id="canvasContainer">

<canvas id="teachCanvas"></canvas>

</div>

<br>

<label>
Object class:
</label>

<br>

<select id="classSelect">

{% for id, name in classes.items() %}

<option value="{{ id }}">{{ name }}</option>

{% endfor %}

</select>

<br>

<button id="saveButton" onclick="saveExample()">
Save Example
</button>

<br>

<button id="cancelButton" onclick="cancelTeaching()">
Cancel
</button>

<div class="status" id="status"></div>

</div>

</div>


<script>

let canvas;
let ctx;

let drawing = false;

let startX = 0;
let startY = 0;

let currentX = 0;
let currentY = 0;

let imageWidth = 0;
let imageHeight = 0;

let capturedFilename = null;


// ==========================================================
// START TEACHING
// ==========================================================

async function startTeaching() {

    const response = await fetch("/capture");

    const data = await response.json();

    if (!data.success) {

        alert("Failed to capture frame");

        return;
    }

    capturedFilename = data.filename;

    document.getElementById("teachPanel").style.display = "block";

    canvas = document.getElementById("teachCanvas");

    const image = new Image();

    image.onload = function() {

        imageWidth = image.naturalWidth;
        imageHeight = image.naturalHeight;

        canvas.width = imageWidth;
        canvas.height = imageHeight;

        ctx = canvas.getContext("2d");

        ctx.drawImage(
            image,
            0,
            0,
            imageWidth,
            imageHeight
        );

        setupDrawing();

    };

    image.src = "/pending/" + capturedFilename;
}


// ==========================================================
// DRAWING
// ==========================================================

function setupDrawing() {

    canvas.onmousedown = function(event) {

        const rect = canvas.getBoundingClientRect();

        startX = (event.clientX - rect.left)
                 * (canvas.width / rect.width);

        startY = (event.clientY - rect.top)
                 * (canvas.height / rect.height);

        drawing = true;

    };


    canvas.onmousemove = function(event) {

        if (!drawing)
            return;

        const rect = canvas.getBoundingClientRect();

        currentX = (event.clientX - rect.left)
                   * (canvas.width / rect.width);

        currentY = (event.clientY - rect.top)
                   * (canvas.height / rect.height);

        redraw();

    };


    canvas.onmouseup = function() {

        drawing = false;

    };

}


// ==========================================================
// REDRAW
// ==========================================================

function redraw() {

    const image = new Image();

    image.onload = function() {

        ctx.drawImage(
            image,
            0,
            0,
            canvas.width,
            canvas.height
        );

        const x = Math.min(startX, currentX);
        const y = Math.min(startY, currentY);

        const width = Math.abs(currentX - startX);
        const height = Math.abs(currentY - startY);

        ctx.strokeStyle = "#00ff00";
        ctx.lineWidth = 3;

        ctx.strokeRect(
            x,
            y,
            width,
            height
        );

    };

    image.src = "/pending/" + capturedFilename;

}


// ==========================================================
// SAVE
// ==========================================================

async function saveExample() {

    const x = Math.min(startX, currentX);
    const y = Math.min(startY, currentY);

    const width = Math.abs(currentX - startX);
    const height = Math.abs(currentY - startY);

    if (width < 5 || height < 5) {

        alert("Draw a bounding box first.");

        return;
    }

    const classId =
        parseInt(
            document.getElementById("classSelect").value
        );

    const payload = {

        filename: capturedFilename,

        class_id: classId,

        x: x,
        y: y,

        width: width,
        height: height,

        image_width: imageWidth,
        image_height: imageHeight
    };


    const response = await fetch(
        "/save_example",
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)
        }
    );


    const result = await response.json();


    if (result.success) {

        document.getElementById("status").innerText =
            "Example saved successfully!";

        setTimeout(
            cancelTeaching,
            1000
        );

    } else {

        document.getElementById("status").innerText =
            "Error: " + result.error;

    }

}


// ==========================================================
// CANCEL
// ==========================================================

function cancelTeaching() {

    document.getElementById("teachPanel").style.display = "none";

    if (capturedFilename) {

        fetch(
            "/cancel/" + capturedFilename,
            {
                method: "DELETE"
            }
        );

    }

    capturedFilename = null;

}

</script>

</body>

</html>
"""

    return render_template_string(
        html,
        classes=CLASS_NAMES
    )


# ============================================================
# VIDEO
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# CAPTURE CURRENT FRAME
# ============================================================

@app.route("/capture")
def capture():

    with frame_lock:

        if latest_frame is None:

            return jsonify({
                "success": False
            })

        frame = latest_frame.copy()


    filename = (
        "teach_"
        + str(int(time.time() * 1000))
        + ".jpg"
    )

    path = os.path.join(
        PENDING_DIR,
        filename
    )

    cv2.imwrite(
        path,
        frame
    )

    return jsonify({
        "success": True,
        "filename": filename
    })


# ============================================================
# SERVE PENDING IMAGE
# ============================================================

@app.route("/pending/<filename>")
def pending(filename):

    return send_from_directory(
        PENDING_DIR,
        filename
    )


# ============================================================
# SAVE TRAINING EXAMPLE
# ============================================================

@app.route("/save_example", methods=["POST"])
def save_example():

    data = request.json

    filename = data["filename"]

    class_id = int(data["class_id"])

    x = float(data["x"])
    y = float(data["y"])

    width = float(data["width"])
    height = float(data["height"])

    image_width = float(data["image_width"])
    image_height = float(data["image_height"])


    # --------------------------------------------------------
    # Convert pixel coordinates to YOLO format
    # --------------------------------------------------------

    center_x = x + width / 2
    center_y = y + height / 2


    center_x /= image_width
    center_y /= image_height

    width /= image_width
    height /= image_height


    # --------------------------------------------------------
    # Move image
    # --------------------------------------------------------

    source = os.path.join(
        PENDING_DIR,
        filename
    )

    destination = os.path.join(
        IMAGE_DIR,
        filename
    )

    if not os.path.exists(source):

        return jsonify({
            "success": False,
            "error": "Captured image not found"
        })


    shutil.move(
        source,
        destination
    )


    # --------------------------------------------------------
    # Write YOLO annotation
    # --------------------------------------------------------

    label_filename = (
        os.path.splitext(filename)[0]
        + ".txt"
    )

    label_path = os.path.join(
        LABEL_DIR,
        label_filename
    )


    with open(label_path, "w") as f:

        f.write(
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{width:.6f} "
            f"{height:.6f}\n"
        )


    print(
        f"TEACH: {filename} -> "
        f"{CLASS_NAMES[class_id]}"
    )


    return jsonify({
        "success": True
    })


# ============================================================
# CANCEL CAPTURE
# ============================================================

@app.route("/cancel/<filename>", methods=["DELETE"])
def cancel(filename):

    path = os.path.join(
        PENDING_DIR,
        filename
    )

    if os.path.exists(path):

        os.remove(path)

    return jsonify({
        "success": True
    })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print(" Robotic Arm YOLO Teach Dashboard")
    print("======================================")
    print()
    print("Teach dataset:")
    print(TEACH_DIR)
    print()
    print("Starting server...")
    print()

    try:

        app.run(
            host="0.0.0.0",
            port=5000,
            threaded=True
        )

    finally:

        camera.release()
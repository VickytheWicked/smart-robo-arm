from flask import Flask, Response, render_template_string
import cv2

app = Flask(__name__)

# Open the USB webcam
camera = cv2.VideoCapture(0)

# Check whether camera opened
if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit(1)

print("Webcam opened successfully.")


def generate_frames():
    while True:
        # Capture frame
        success, frame = camera.read()

        if not success:
            print("ERROR: Failed to read frame")
            break

        # Encode frame as JPEG
        success, buffer = cv2.imencode(
            '.jpg',
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not success:
            continue

        # Convert JPEG to bytes
        frame_bytes = buffer.tobytes()

        # Send frame as MJPEG stream
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame_bytes +
            b'\r\n'
        )


@app.route('/')
def index():

    html = """
    <!DOCTYPE html>
    <html>

    <head>
        <title>Raspberry Pi Camera</title>

        <style>
            body {
                background: #111;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
            }

            h1 {
                margin-top: 30px;
            }

            img {
                width: 80%;
                max-width: 1000px;
                border: 3px solid white;
                border-radius: 10px;
                margin-top: 20px;
            }
        </style>

    </head>

    <body>

        <h1>Raspberry Pi 5 Webcam</h1>

        <img src="/video_feed">

    </body>

    </html>
    """

    return render_template_string(html)


@app.route('/video_feed')
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':

    try:
        print("Starting camera server...")
        print("Open the Raspberry Pi IP address in your browser.")

        app.run(
            host='0.0.0.0',
            port=5000,
            threaded=True
        )

    finally:
        camera.release()
# 🤖 6-DOF Smart Robotic Arm — Autonomous Vision & Voice Control

A multi-tiered, vision-guided 6-DOF robotic arm manipulation system combining **embedded microcontroller control**, **dual-path computer vision (Local HSV + Gemini Robotics VLM)**, **real-time visual servoing (OpenCV CSRT)**, **multimodal voice interaction ("Lucy" powered by Gemini Live)**, and a **responsive web dashboard**.

The system enables autonomous detection, continuous visual tracking, alignment, and grasping of objects, as well as manual joint/directional manipulation and voice-commanded task execution.

---

## 📸 System Overview & Dashboard

The web dashboard acts as a mission control center hosted directly by the robot control server (`server_v9.py`):

* 🎛️ **Manual Joint Sliders**: Direct angle control for all 6 servos with software limits.
* 🦾 **Real-Time 2D Arm Kinematics Visualizer**: Interactive dynamic SVG preview reflecting joint poses.
* 📷 **Low-Latency Camera HUD**: Live MJPEG video feed overlaid with crosshairs, bounding boxes, centroid tracking, and pixel errors.
* 🎯 **Dual-Path Object Picking**:
  * **Color Detection (HSV)**: Ultra-fast (~30 FPS) local tracking for colored items (red, green, yellow, blue, white).
  * **Universal Object Picking**: Zero-shot open-vocabulary localization via `gemini-robotics-er-2-preview` with anti-drift CSRT tracking.
* 🎙️ **"Lucy" Gemini Live Voice Interface**: Direct conversational interaction with natural language robot control tool execution.
* 🕹️ **Directional Manual D-Pad**: Stepwise navigation (`LEFT`, `RIGHT`, `FORWARD`, `BACK`, `CENTERED`).
* 📜 **Live Telemetry & Logs**: Real-time Socket.IO status stream, servo angles, and ESP32 serial logs.

---

## 🏗️ System Architecture

```text
                                  ┌────────────────────────────────┐
                                  │   User / Operator Interface    │
                                  ├────────────────────────────────┤
                                  │  Web Dashboard (v3) / Voice    │
                                  └──────────────┬─────────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   │ HTTP / Socket.IO (Port 3000)                              │ WebSocket (Port 8000)
                   ▼                                                           ▼
    ┌──────────────────────────────────────────────┐            ┌──────────────────────────────────────────────┐
    │          Robot Server (server_v9.py)         │            │      Gemini Live Server (gemini_server_v2)   │
    ├──────────────────────────────────────────────┤            ├──────────────────────────────────────────────┤
    │ • Flask + Socket.IO Control Engine           │◄──HTTP────┤ • FastAPI + WebSockets Gateway               │
    │ • Visual Servoing Closed-Loop Controller     │  (Arm V2)  │ • "Lucy" Multimodal Voice Agent              │
    │ • Video Stream (MJPEG + HUD Annotations)     │            │ • Gemini Live API (gemini-3.1-flash-live)    │
    │ • Serial Protocol Bridge (115200 baud)       │            │ • Function Calling Tool Mapping (tools_v2)   │
    └──────────────┬───────────────────────────────┘            └──────────────────────────────────────────────┘
                   │
         ┌─────────┴────────────────────────────────┐
         │ Perception & Tracking Subsystems         │
         ├──────────────────────────────────────────┤
         │ [Path A] Fast Local HSV (vision_detection) │
         │   • Red, Green, Yellow, Blue, White      │
         │   • Morphological Filtering & Centroid   │
         │                                          │
         │ [Path B] Universal Target (crst.py logic)│
         │   • Gemini Robotics ER-2 VLM             │
         │   • OpenCV CSRT Tracker @ 30 FPS         │
         │   • Anti-Drift Heuristics & Re-query     │
         └─────────────────┬────────────────────────┘
                           │
                 Commands: S<idx>:<deg> / Directional
                           │
                           ▼
    ┌──────────────────────────────────────────────┐
    │     Embedded Controller (CameraEquipedRobo)  │
    ├──────────────────────────────────────────────┤
    │ • Seeed Studio XIAO ESP32-C3 Firmware        │
    │ • PCA9685 16-Channel PWM Servo Driver        │
    │ • Table Collision Boundary Protection        │
    │ • Trajectory Sequences & Safety Limits       │
    └──────────────────────┬───────────────────────┘
                           │ PWM Pulses
                           ▼
    ┌──────────────────────────────────────────────┐
    │             6-DOF Physical Arm               │
    ├──────────────────────────────────────────────┤
    │  J0: Base (Yaw)        J3: Wrist (Pitch)     │
    │  J1: Shoulder (Pitch)  J4: Wrist Roll (Roll) │
    │  J2: Elbow (Pitch)     J5: Gripper           │
    └──────────────────────────────────────────────┘
```

---

## 📂 Core Codebase Structure

| File / Directory | Description |
| :--- | :--- |
| **`server_v9.py`** | **Primary Robot Control Server** (Flask + Socket.IO). Orchestrates camera streaming, visual servoing loop, serial communication with ESP32, REST endpoints, and hosts `dashboard_mobile_v3.html`. |
| **`crst.py`** | **Standalone Universal Visual Servoing Engine**. Demonstrates zero-shot object detection using `gemini-robotics-er-2-preview` paired with high-rate OpenCV CSRT tracking, anti-drift heuristics, and proportional visual servoing. |
| **`vision_detection.py`** | **High-Speed HSV Color Detection Module**. Real-time contour analysis and centroid localization for red, green, yellow, blue, and white objects. Includes interactive calibration trackbar tuner (`--tune`). |
| **`gemini_server_v2.py`** | **Gemini Live Gateway** (FastAPI). Bridges bidirectional browser audio/video streams over WebSocket to Gemini Live (`gemini-3.1-flash-live-preview`), handling speech-to-speech and tool calls. |
| **`gemini_live_v2.py`** | **"Lucy" AI Brain Core**. Implements the conversational agent, system prompt, long-term memory handling, and robot tool invocation. |
| **`robot/arm_v2.py`** | **Robot Arm Python Client Library**. Encapsulates REST API commands (`pick_color_object`, `pick_object`, `move_arm`, `move_joint`, `reset_arm`, `gripper`) for programmatic control. |
| **`robot/tools_v2.py`** | **Gemini Function Calling Definitions**. Tool schemas passed to Gemini Live so Lucy can interpret user intent and invoke arm actions. |
| **`dashboard_mobile_v3.html`** | **Comprehensive Mobile/Desktop Web Dashboard**. Features real-time arm SVG visualizer, camera HUD, manual sliders, D-pad, Gemini Live voice chat controls, and system telemetry. |
| **`CameraEquipedRobo/`** | **PlatformIO C++ Project for XIAO ESP32-C3**. Contains `src/main.cpp` for PCA9685 PWM driving, table safety checks (`TABLE_POINTS`), motion smoothing, and serial command decoding. |
| **`capture.py`** | **Dataset Collection Tool**. Spacebar-triggered camera frame capture utility for training custom object detection models. |
| **`twilio_handler.py`** | **Twilio Media Stream Adapter**. Provides phone-call audio bridge to Gemini Live with 24kHz/16kHz to 8kHz $\mu$-law transcoding. |
| **`old_version/`** | **Historical Development Versions**. Archived previous iterations (`server_v5.py` - `server_v8.py`, homography calibration, depth grasping). |

---

## 🎯 Dual-Mode Perception & Visual Servoing

### 1. High-Speed Local Color Detection (`vision_detection.py`)
* **Execution Rate**: Native ~30 FPS camera framerate (command interval: `0.2s`).
* **Supported Targets**: `red`, `green`, `yellow`, `blue`, `white` (e.g. Lego bricks, erasers, markers).
* **Technique**: Dual-range HSV filtering (handling hue wrap-around for reds), Gaussian smoothing, morphological opening/dilation, and contour moments for centroid $(c_x, c_y)$ computation.
* **Auto-Routing**: If the user or voice agent requests a colored object (e.g. *"Pick up the red lego brick"*), the system automatically routes detection locally, bypassing VLM latency.

### 2. Universal Target Localization (`gemini-robotics-er-2-preview` + CSRT)
* **Execution Rate**: VLM query every `~1.3s - 1.5s`; local CSRT tracking at 30 FPS.
* **Workflow**:
  1. Operator specifies any arbitrary target (e.g., *"white correction tape"*, *"pair of scissors"*).
  2. The frame is dispatched to `gemini-robotics-er-2-preview` to obtain normalized $[y_{min}, x_{min}, y_{max}, x_{max}]$ bounding coordinates.
  3. The bounding box initializes an **OpenCV CSRT tracker**.
  4. **Anti-Drift Validation (`validate_tracker_update`)**: Protects against tracking drift across frames:
     * **Centroid Jump**: Rejects displacements $> 45\text{ px}$.
     * **Area Ratio**: Rejects sudden expansions/contractions $> 2.5\times$.
     * **Aspect Ratio**: Rejects distortion changes $> 1.35\times$.
     * Reverts to VLM re-acquisition if 3 consecutive frames fail.

### 3. Closed-Loop Proportional Visual Servoing
* **Error Formulation**:
  $$\text{err}_x = c_x - \text{center}_x, \quad \text{err}_y = c_y - \text{center}_y$$
* **Control Deadzones**: $60\text{ px} \times 60\text{ px}$ deadzone around frame center.
* **Action Policy**:
  * If $|\text{err}_x| > \text{DEAD\_ZONE}_x$: Arm generates `LEFT:<speed>` or `RIGHT:<speed>`.
  * If $|\text{err}_y| > \text{DEAD\_ZONE}_y$: Arm generates `FORWARD:<speed>` or `BACK:<speed>`.
  * When aligned within deadzone: Sends `CENTERED:<speed>` $\to$ arm moves downward toward target until reaching table grasp threshold $\to$ triggers `GRAB` routine.

---

## 🦾 Hardware & Servo Configuration

The physical arm uses 6 high-torque servos wired to a **PCA9685 16-channel 12-bit I2C PWM driver** controlled by a **Seeed XIAO ESP32-C3**:

| Joint | Servo Index | Name | Motion Type | PWM Pulse Min | PWM Pulse Max | Home Angle |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **J0** | Channel 0 | Base | Yaw (Left/Right) | 85 | 540 | $30^\circ$ |
| **J1** | Channel 1 | Shoulder | Pitch (Up/Down) | 90 | 520 | $180^\circ$ |
| **J2** | Channel 2 | Elbow | Pitch (Extension) | 190 | 520 | $0^\circ$ |
| **J3** | Channel 3 | Wrist | Pitch | 100 | 400 | $40^\circ$ |
| **J4** | Channel 4 | Wrist Roll | Roll (Orientation) | 100 | 550 | $90^\circ$ |
| **J5** | Channel 5 | Gripper | Open / Close | 80 | 400 | $0^\circ$ (Open) |

### Table Collision Protection
The ESP32 firmware features hard-coded boundary protection (`TABLE_POINTS`) mapping elbow vs shoulder limits to prevent the end-effector from colliding with or pressing into the workspace surface during downward visual servoing.

---

## 🚀 Getting Started

### 1. Hardware Setup & Connections
1. Connect the 6-DOF robotic arm servos to PCA9685 channels 0 through 5.
2. Connect PCA9685 I2C lines (`SDA`, `SCL`) and power to the XIAO ESP32-C3.
3. Plug the ESP32 into your host computer/Raspberry Pi via USB (e.g. `/dev/ttyACM0`).
4. Mount the camera (e.g. Logitech webcam) onto the arm or workspace with an overhead/end-effector perspective.

### 2. Flash ESP32 Firmware
```bash
cd CameraEquipedRobo
# Build and upload firmware using PlatformIO
pio run --target upload
```

### 3. Environment Setup
Install required Python dependencies:
```bash
pip install -r requirements.txt
```

Create or configure `.env` file in the project root:
```env
GEMINI_API_KEY="your_google_gemini_api_key"
SERIAL_PORT="/dev/ttyACM0"
BAUD_RATE=115200
HOST="0.0.0.0"
PORT=3000
CAMERA_INDEX=1
COMMAND_INTERVAL=0.6
COLOR_COMMAND_INTERVAL=0.2
ROBOT_SERVER_URL="http://127.0.0.1:3000"
```

---

## 🏃 Running the Application

### Option A: Complete Web & Voice Experience (Recommended)

1. **Start the Main Robot Server (`server_v9.py`)**:
   ```bash
   python server_v9.py
   ```
   * Starts the Flask + Socket.IO server on `http://localhost:3000`.
   * Connects to the ESP32 on `/dev/ttyACM0` (or runs simulated in `DRY_RUN` mode).
   * Serves `dashboard_mobile_v3.html` and the `/video_feed` MJPEG stream.

2. **Start the Gemini Live Voice Agent (`gemini_server_v2.py`)**:
   ```bash
   python gemini_server_v2.py
   ```
   * Launches FastAPI on `http://localhost:8000`.
   * Accepts WebSocket connections on `ws://localhost:8000/ws` for real-time audio/vision conversation with "Lucy".

3. **Open the Dashboard**:
   * Navigate to `http://localhost:3000` in your web browser.
   * Click **CONNECT** under the Lucy Voice Assistant panel to enable bidirectional voice communication.

---

### Option B: Standalone Universal Visual Servoing (`crst.py`)

To run dedicated visual servoing without launching the full voice server:
```bash
python crst.py
```
* Queries `gemini-robotics-er-2-preview` for the target object configured in `TARGET`.
* Tracks continuously with OpenCV CSRT at 30 FPS and sends real-time servo commands directly over serial.

---

### Option C: HSV Color Detector & Tuner (`vision_detection.py`)

To test or recalibrate HSV color thresholds under your current ambient lighting:
```bash
# Interactive trackbar calibration tool:
python vision_detection.py --tune

# Run single-camera color test:
python vision_detection.py
```

---

## 📡 REST API & Socket.IO Reference

### REST Endpoints (`server_v9.py`)

| Method | Route | Description | Payload Example |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Web control dashboard (`dashboard_mobile_v3.html`) | — |
| `GET` | `/video_feed` | Live MJPEG video stream with tracking HUD overlays | — |
| `GET` | `/api/status` | Current servo angles, tracking state, and connection status | — |
| `POST` | `/api/pick` | Activate visual servoing for a target or color | `{"target": "red block", "color": "red"}` |
| `POST` | `/api/stop-tracking` | Immediately stops visual servoing and tracking | `{}` |
| `POST` | `/api/command` | Send directional command to ESP32 | `{"command": "LEFT", "speed": 8}` |
| `POST` | `/api/servo` | Set specific joint angle | `{"servo": 0, "angle": 45}` |
| `POST` | `/api/gripper/<action>` | Open or close gripper end-effector | `<action>`: `open` or `close` |
| `POST` | `/api/reset` | Move arm to default home calibration pose | `{}` |
| `POST` | `/api/mode` | Switch control mode (`manual`, `visual`, `scripted`) | `{"mode": "visual"}` |

### Socket.IO Telemetry Events
* **`state_update`**: Real-time broadcast of servo positions, tracking status, and coordinate errors.
* **`log_entry`**: Streaming log messages from the visual servo controller and ESP32 serial replies.
* **`mode_change`** & **`target_change`**: Status synchronization across all connected browser clients.

---

## 🎙️ Voice Commands ("Lucy") Examples

When connected via the dashboard or WebSocket, you can speak directly to Lucy:

* 🗣️ *"Lucy, pick up the red lego brick."* $\to$ Automatically invokes `pick_color_object(color="red", item="lego brick")`.
* 🗣️ *"Grab the yellow block on the table."* $\to$ Automatically invokes `pick_color_object(color="yellow", item="block")`.
* 🗣️ *"Can you locate and grab the black scissors?"* $\to$ Invokes `pick_object(target="black scissors")` using Gemini Robotics ER-2.
* 🗣️ *"Stop tracking immediately."* $\to$ Executes emergency safety stop `stop_tracking()`.
* 🗣️ *"Open the gripper and reset the arm."* $\to$ Executes `open_gripper()` then `reset_arm()`.

---

## 🛡️ Safety & Operational Guidelines

1. **Dry Run Mode**: Set `DRY_RUN=True` in `.env` to test camera tracking, Gemini localization, and visual servo math without sending physical motion commands to the servos.
2. **Emergency Stop**: The **Reset** button on the dashboard or the *"Stop tracking"* voice command immediately disengages the visual servoing loop and halts arm motion.
3. **Power Supply**: Ensure the PCA9685 servo driver is powered by an independent external 5V/6V high-current power supply (minimum 4A–6A recommended for 6 servos under load) rather than drawing power directly from the microcontroller or Raspberry Pi USB rails.

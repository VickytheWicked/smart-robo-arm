# 🤖 Robotic Arm — Vision-Based Control & Visual Servoing

A **6-DOF robotic arm project** combining embedded control, computer vision, and a web-based robotics dashboard. The system uses a **Raspberry Pi as the main host**, an **XIAO ESP32-C3 for servo control**, a **PCA9685 servo driver**, and a **Logitech webcam** for visual feedback.

The long-term goal is to develop a robotic arm capable of **detecting, tracking, approaching, and grasping objects autonomously using vision-based control**.

> 🚧 **Project Status: Under Development**
>
> The current visual servoing system is primarily **rule-based**. It is being developed toward more robust vision-based control methods such as **Image-Based Visual Servoing (IBVS)**.

---

## 📸 Dashboard

The Raspberry Pi hosts a web server that provides a browser-based control and monitoring interface.

![Robotic Arm Dashboard](dashboard.png)

The dashboard provides:

* 🎛️ Manual joint control
* 🦾 Real-time arm visualization
* 📷 Live camera feed
* 🎯 Visual servoing
* 📏 Sensor information
* 🧠 Autonomous agent interface
* 📜 System and exploration logs
* ⚙️ Scripted arm movements
* 🔌 ESP32 communication status

## Screenshot
![Dashboard](/images/dashboard.png)

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │      Logitech        │
                    │       Webcam         │
                    └──────────┬───────────┘
                               │
                               │ Video
                               ▼
                    ┌──────────────────────┐
                    │    Raspberry Pi      │
                    │                      │
                    │  Web Server           │
                    │  Computer Vision      │
                    │  Visual Servoing      │
                    │  Dashboard            │
                    └──────────┬───────────┘
                               │
                         Commands / Control
                               │
                               ▼
                    ┌──────────────────────┐
                    │    XIAO ESP32-C3     │
                    │                      │
                    │  Servo Controller    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      PCA9685         │
                    │   PWM Servo Driver   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     6-DOF Robotic    │
                    │         Arm          │
                    └──────────────────────┘
```

---

## 🔧 Hardware

| Component            | Role                                                                |
| -------------------- | ------------------------------------------------------------------- |
| 🖥️ Raspberry Pi     | Main computer, web server, vision processing and high-level control |
| 🔌 XIAO ESP32-C3     | Low-level servo controller                                          |
| 🎛️ PCA9685          | Multi-channel PWM servo control                                     |
| 🦾 6-DOF Robotic Arm | Physical manipulation platform                                      |
| 📷 Logitech Webcam   | Object detection and visual feedback                                |

---

## 🌐 Web-Based Control System

The Raspberry Pi hosts a web server that provides a centralized browser-based interface for controlling and monitoring the robotic arm.

### Manual Control

The dashboard provides individual controls for:

```text
J0 BASE
J1 SHOULDER
J2 ELBOW
J3 WRIST
J4 WRIST ROLL
J5 GRIPPER
```

Users can adjust joint positions and send the resulting commands to the ESP32.

### Control Modes

**Manual**

```text
User → Dashboard → Raspberry Pi → ESP32 → PCA9685 → Servos
```

**Scripted**

Predefined sequences can be executed for actions such as:

* Home
* Move
* Lift
* Place
* Pick
* Gripper control

**Visual Servoing**

The vision system determines movement corrections based on the camera feed.

---

# 📷 Camera & Computer Vision

A Logitech webcam provides the visual input for the vision subsystem.

The Raspberry Pi receives the camera stream and processes the visual information to determine where the target object is located.

```text
Camera
   │
   ▼
Image Processing
   │
   ▼
Object Detection / Tracking
   │
   ▼
Target Position
   │
   ▼
Visual Servo Controller
   │
   ▼
Robot Motion
```

---

# 🎯 Visual Servoing

One of the main objectives of this project is **vision-based autonomous manipulation**.

The current system attempts to:

1. Detect an object using the camera
2. Determine its position in the image
3. Calculate an appropriate movement
4. Move the robotic arm
5. Continuously observe the camera
6. Correct the arm's position
7. Approach the object
8. Attempt to grasp it

### Current Approach

The current implementation uses a **rule-based controller**.

For example:

```text
IF object is left of image center
    → move arm left

IF object is right of image center
    → move arm right

IF object is above target region
    → adjust arm upward

IF object is below target region
    → adjust arm downward

IF object is sufficiently aligned
    → attempt grasp
```

This approach is useful for prototyping the complete **perception → control → actuation** pipeline, but it has limitations in precision, generalization, and stability.

---

# 🧠 Future: Image-Based Visual Servoing (IBVS)

The next major development direction is implementing a more principled **Image-Based Visual Servoing (IBVS)** system.

Instead of relying primarily on predefined rules, IBVS can use the relationship between image features and robot motion.

```text
Image Features
      ↓
Image Error
      ↓
Interaction / Image Jacobian
      ↓
Camera / End-Effector Velocity
      ↓
Robot Motion
```

The objective is to minimize the visual error between the **current image feature** and the **desired image feature**.

```text
        Desired Feature
              ●
              │
              │  Image Error
              │
              ▼
        Current Feature
              ●
              │
              ▼
      Visual Servo Controller
              │
              ▼
         Robot Motion
              │
              ▼
          New Image
              │
              └───────────────┐
                              │
                              ▼
                     Repeat / Converge
```

IBVS is **not currently the primary controller**. It is a planned direction for improving the visual servoing subsystem.

---

# 🔄 Overall Control Loop

The intended autonomous control architecture is:

```text
             ┌───────────────┐
             │    Camera     │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ Object /      │
             │ Feature       │
             │ Detection     │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ Visual Error  │
             │ Calculation   │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ Visual Servo  │
             │ Controller    │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ Robot Command │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │    ESP32-C3   │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │   PCA9685     │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │  Robotic Arm  │
             └───────┬───────┘
                     │
                     ▼
                   Camera
                     │
                     └──────────────► Feedback
```

---

# 🛠️ Technology Stack

### Hardware

* Raspberry Pi
* XIAO ESP32-C3
* PCA9685 16-channel PWM driver
* 6-DOF robotic arm
* Servo motors
* Logitech webcam

### Software

* Python
* OpenCV
* Web server
* HTML / CSS / JavaScript
* ESP32 firmware
* Serial / network-based communication
* Computer vision
* Visual servoing

---

# 🚧 Current Development Status

| Feature                      | Status        |
| ---------------------------- | ------------- |
| ESP32 servo control          | ✅ Working     |
| PCA9685 servo control        | ✅ Working     |
| 6-DOF arm control            | ✅ Working     |
| Raspberry Pi control server  | ✅ Working     |
| Web dashboard                | ✅ Working     |
| Manual joint control         | ✅ Working     |
| Camera integration           | ✅ Working     |
| Camera feed on dashboard     | ✅ Working     |
| Scripted movements           | 🔄 Developing |
| Object tracking              | 🔄 Developing |
| Rule-based visual servoing   | 🔄 Developing |
| Autonomous grasping          | 🔄 Developing |
| IBVS controller              | 🧪 Planned    |
| Robust object detection      | 🧪 Planned    |
| Full autonomous manipulation | 🧪 Planned    |

---

# 🎯 Project Goal

The ultimate goal is to evolve the system from a remotely controlled robotic arm into a **closed-loop vision-based manipulation platform**.

```text
Manual Control
      ↓
Scripted Control
      ↓
Rule-Based Vision
      ↓
Closed-Loop Visual Servoing
      ↓
IBVS
      ↓
Autonomous Pick & Place
```

The project is being developed incrementally, starting with reliable hardware control and gradually introducing more advanced robotics and computer vision algorithms.

---

# 📌 Project Status

**Active Development 🚧**

This project explores the intersection of:

* Robotics
* Computer Vision
* Embedded Systems
* Web-Based Robot Control
* Autonomous Manipulation
* Visual Servoing
* Image-Based Visual Servoing (IBVS)

The visual servoing subsystem is still experimental, and the current implementation should be considered a **prototype rather than a production-grade autonomous manipulation system**.

---

## ⭐ Future Vision

> **See → Understand → Move → Correct → Grasp**

The long-term objective is to build a robotic system that can use visual feedback to continuously correct its own movements rather than simply executing predefined joint commands.

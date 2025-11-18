# Finger Counter

A small Python script that uses MediaPipe Hands and OpenCV to detect hands from a webcam, count raised fingers, and draw simple UI panels in real time.

## Features
- Tracks up to **two hands**
- Counts raised fingers using landmark positions
- Displays handedness, finger count, detection score, and tracking info
- Shows coordinate labels for each landmark
- Clean UI panels on-screen for clarity
- **"x" on the keyboard to close window**

## Requirements
Python **3.11.0** is the version of Python the project was created on, but if MediaPipe fails to install on Windows, use **Python 3.10.0**.

Install dependencies (on the assumption interpreter is 3.11.0):
```bash
pip install opencv-python numpy mediapipe
```

Configuration

You can adjust settings at the top of the script:

```MAX_HANDS```

```DETECT_CONF``` / ```TRACK_CONF```

```HAND_COLORS```

Camera resolution (```CAP_PROP_FRAME_WIDTH``` / ```HEIGHT```) <- recomended resolution 1920x1080

## Troubleshooting

- If MediaPipe refuses to install, switch to Python 3.10.

- Lower camera resolution if the program feels slow.


11/17/25

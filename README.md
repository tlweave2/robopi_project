# Eddie Haskell Robot

A Raspberry Pi robot that finds faces, recognizes who it is looking at and follows them around. I designed and 3D printed the chassis and wrote the vision and motor control software. It started as a weekend project and it got out of hand.

The name comes from the Leave It to Beaver character. Eddie is extra polite to the people he knows and pretty indifferent to everyone else.

## What it does

- Detects faces in real time on the Pi with MediaPipe BlazeFace at 640×480.
- Sends each face crop to a recognition server on my laptop and gets back a name and a confidence score.
- Works out where the person is (LEFT, CENTER or RIGHT) and how close they are (NEAR, MEDIUM or FAR) from the face box.
- Turns toward the person and drives closer when they are far away.
- Picks a greeting based on how confident the match is.

## How it works

The work is split between the Pi and a desktop machine on the same network. The Pi handles the fast work that has to happen on the robot. The desktop handles the heavy recognition model.

```
 Camera ─► Pi agent (pi_agent/pi_agent.py)
            │  MediaPipe face detection
            │  position + distance estimate
            │  160×160 face crop
            ▼
        HTTP POST /recognize  (~100 ms over LAN, 0.5 s timeout)
            ▼
        Desktop server (desktop_server/server.py)
            │  InsightFace embeddings
            │  cosine similarity against an NPZ face gallery
            ▼
        { name, confidence } ─► Pi agent ─► motor control (pigpio PWM)
```

**Pi agent**
- `VisionTools` runs detection and builds an observer state string like `Tim CENTER NEAR (0.86)` that the decision loop reads.
- `MotorControl` drives two encoder motors with pigpio PWM. The left and right speeds are calibrated separately so it drives straight. Every move is safety clamped (90° max turn and 2 s max move) and the motors always stop on exit.
- `RoboPiAgent` runs the follow loop at about 2 Hz. It has a milestone test mode that logs detections and FPS.
- If the desktop server cannot be reached the agent falls back to "Unknown" and keeps running.

**Desktop server**
- A Flask API with `/recognize`, `/enroll`, `/gallery`, `/backup` and `/health`.
- It uses InsightFace when it is installed and falls back to OpenCV LBPH if it is not.
- Enrolled faces are saved to an NPZ gallery and backed up on every enroll.

## Hardware

- Raspberry Pi 4 (8GB)
- GoPro Hero 7 as the camera
- 2 DC motors with encoders and a dual motor driver
- Ryobi 18V 5Ah drill battery for power. It lasts a long time.
- USB mic, Bluetooth speaker and a small monitor for debugging
- 1TB external drive for storage
- Custom chassis, battery adapters, Pi case and mounts that I designed and 3D printed. The STL files are in [`robot/`](robot/).

## Running it

Desktop side:

```bash
pip install flask opencv-contrib-python numpy insightface
python desktop_server/server.py        # listens on port 8080
```

Pi side (needs the pigpio daemon running):

```bash
sudo pigpiod
pip install opencv-python mediapipe numpy requests pigpio
python pi_agent/pi_agent.py --mode milestone --duration 30   # detection test
python pi_agent/pi_agent.py --mode follow --duration 60      # follow a person
```

Set `desktop_server_url` in `config/vision_config.json` to your desktop's IP address. The config files are created with default values on the first run.

## What I would do next

- Use the wheel encoders for closed loop turns instead of timed turns.
- Add a search behavior when no one is in view.
- Hook the greetings up to text to speech on the speaker.
- Move recognition onto the Pi with a smaller model so it can run without the desktop.

It works pretty well as long as the lighting is decent.

# Eddie Haskell Robot

My Raspberry Pi robot that follows people around and is extra nice to my girlfriend Sara Jane.

## What It Does

- Sees faces with the GoPro and follows you around
- Super polite to Sara Jane, normal with everyone else
- Built from stuff I had laying around plus some motor parts

## Parts List

- Pi 4 8GB 
- Old WD Passport drive (1TB)
- GoPro Hero 7 (borrowed permanently)
- USB mic that was in a drawer
- Bluetooth speaker
- Ryobi drill battery (18V 5Ah - lasts forever)
- 2 motors with encoders from Aliexpress
- Random mini monitor for debugging
- **All chassis/mounts/brackets 3D designed and printed by me**

## Getting It Running

Laptop side:
```bash
git clone https://github.com/tlweave2/robopi_project.git
pip install flask opencv-contrib-python numpy insightface
python desktop_server/src/recognition_server.py
```

Pi side:
```bash
# Same repo, then:
pip install opencv-python mediapipe numpy requests pigpio
python pi_agent/src/robopi_agent.py
```

## The Gimmick

When Eddie sees Sara Jane: *"Good afternoon Miss Sara Jane, you look wonderful today!"*

When he sees me: *"Oh hey."*

That's it. Works pretty well when the lighting doesn't suck.

---
*Weekend project that got out of hand*

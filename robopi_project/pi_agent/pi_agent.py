#!/usr/bin/env python3
"""
Pi Agent System - Following Design Docs
- MediaPipe BlazeFace detection (~5-15 FPS at 640×480)
- Hybrid recognition (Pi detection + Desktop recognition)
- Tool API: detect_faces(), identify_person()
- Observer state format: "Dad CENTER NEAR (0.86)"
"""

import cv2
import mediapipe as mp
import numpy as np
import requests
import base64
import json
import time
import logging
import pigpio
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class VisionTools:
    """
    Vision system following design docs:
    - MediaPipe BlazeFace for detection
    - Hybrid Pi+Desktop recognition
    - Tool API compliance
    """
    
    def __init__(self, config_path="config/vision_config.json"):
        self.logger = logging.getLogger(__name__)
        self.config = self.load_config(config_path)
        
        # Initialize MediaPipe Face Detection (BlazeFace)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (2m) as per docs
            min_detection_confidence=self.config["detection_confidence"]
        )
        
        # Desktop server connection
        self.desktop_url = self.config["desktop_server_url"]
        
        # Camera setup
        self.camera = None
        self.init_camera()
        
        self.logger.info("Vision tools initialized per design docs")
    
    def load_config(self, config_path):
        """Load vision configuration"""
        default_config = {
            "detection_confidence": 0.5,
            "recognition_threshold": 0.6,
            "camera_width": 640,
            "camera_height": 480,
            "desktop_server_url": "http://192.168.1.25:8080",
            "face_crop_size": 160,
            "position_threshold": 100,
            "distance_thresholds": {
                "near": 15000,
                "medium": 5000
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            # Create config directory and file
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def init_camera(self):
        """Initialize camera with design doc specs"""
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera_width"])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera_height"])
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify camera
        ret, frame = self.camera.read()
        if not ret:
            self.logger.error("Camera initialization failed")
            raise RuntimeError("Camera not accessible")
        
        self.logger.info(f"Camera initialized: {self.config['camera_width']}x{self.config['camera_height']}")
    
    def detect_faces(self) -> List[Dict[str, Any]]:
        """
        Tool API: detect_faces() -> [{name,confidence,position,distance,box}]
        As specified in design docs for Observer state
        """
        ret, frame = self.camera.read()
        if not ret:
            self.logger.warning("Failed to read camera frame")
            return []
        
        # MediaPipe detection on Pi (fast CPU detection)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                face_data = self._process_detection(detection, frame)
                if face_data:
                    faces.append(face_data)
        
        return faces
    
    def _process_detection(self, detection, frame) -> Optional[Dict[str, Any]]:
        """Process MediaPipe detection into design doc format"""
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        
        # Convert relative coordinates to absolute
        x = max(0, int(bboxC.xmin * w))
        y = max(0, int(bboxC.ymin * h))
        width = min(int(bboxC.width * w), w - x)
        height = min(int(bboxC.height * h), h - y)
        
        if width <= 0 or height <= 0:
            return None
        
        # Calculate position relative to frame center
        center_x = x + width // 2
        frame_center = w // 2
        
        if center_x < frame_center - self.config["position_threshold"]:
            position = "LEFT"
        elif center_x > frame_center + self.config["position_threshold"]:
            position = "RIGHT"
        else:
            position = "CENTER"
        
        # Estimate distance based on face size
        face_size = width * height
        if face_size > self.config["distance_thresholds"]["near"]:
            distance = "NEAR"
        elif face_size > self.config["distance_thresholds"]["medium"]:
            distance = "MEDIUM"
        else:
            distance = "FAR"
        
        # Extract 160×160 face crop for recognition
        face_crop = frame[y:y+height, x:x+width]
        face_160 = cv2.resize(face_crop, (self.config["face_crop_size"], self.config["face_crop_size"]))
        
        # Get recognition from Desktop (hybrid processing)
        recognition = self.identify_person(face_160)
        
        return {
            "name": recognition["name"],
            "confidence": recognition["confidence"],
            "position": position,
            "distance": distance,
            "box": [x, y, width, height],
            "center": (center_x, y + height // 2)
        }
    
    def identify_person(self, face_crop) -> Dict[str, Any]:
        """
        Tool API: identify_person(face_crop) -> {name, confidence}
        Sends 160×160 crop to Desktop, gets recognition back
        """
        try:
            # Encode face crop for transmission
            _, buffer = cv2.imencode('.png', face_crop)
            face_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send to Desktop server (~100ms latency over LAN)
            response = requests.post(
                f"{self.desktop_url}/recognize",
                json={"face_image": face_b64},
                timeout=0.5  # Fast timeout for real-time feel
            )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.debug(f"Desktop recognition: {result}")
                return result
            else:
                self.logger.warning(f"Desktop server error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Desktop connection failed: {e}")
        
        # Fallback for offline operation
        return {"name": "Unknown", "confidence": 0.0}
    
    def get_observer_state(self) -> str:
        """
        Generate Observer state format for Executor/Planner
        Format: "Dad CENTER NEAR (0.86)" or "None detected"
        """
        faces = self.detect_faces()
        
        if faces:
            # Use primary face (highest confidence or first detected)
            primary_face = max(faces, key=lambda f: f['confidence'])
            return f"{primary_face['name']} {primary_face['position']} {primary_face['distance']} ({primary_face['confidence']:.2f})"
        else:
            return "None detected"

class MotorControl:
    """
    Motor control following design doc calibration
    Using actual GPIO pins and calibrated speeds
    """
    
    def __init__(self, config_path="config/motor_config.json"):
        self.logger = logging.getLogger(__name__)
        self.config = self.load_config(config_path)
        
        # Initialize pigpio
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon not running")
        
        self.setup_pins()
        self.logger.info("Motor control initialized with calibrated speeds")
    
    def load_config(self, config_path):
        """Load motor configuration with your calibrated values"""
        default_config = {
            # GPIO pins from your wiring doc
            "L_FWD": 12, "L_REV": 13,
            "R_FWD": 19, "R_REV": 16,
            "L_EN_R": 5, "L_EN_L": 6,
            "R_EN_R": 26, "R_EN_L": 20,
            
            # Your calibrated speeds
            "LEFT_SPEED": 95,
            "RIGHT_SPEED": 110,
            
            # Safety limits
            "max_turn_deg": 90,
            "max_move_s": 2.0,
            "pwm_frequency": 1000
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def setup_pins(self):
        """Setup motor control pins"""
        # PWM pins
        for pin in [self.config["L_FWD"], self.config["L_REV"], 
                   self.config["R_FWD"], self.config["R_REV"]]:
            self.pi.set_mode(pin, pigpio.OUTPUT)
            self.pi.set_PWM_frequency(pin, self.config["pwm_frequency"])
        
        # Enable pins
        for pin in [self.config["L_EN_R"], self.config["L_EN_L"],
                   self.config["R_EN_R"], self.config["R_EN_L"]]:
            self.pi.set_mode(pin, pigpio.OUTPUT)
            self.pi.write(pin, 1)  # Enable motors
    
    def turn_left(self, angle_deg: float):
        """Turn left by specified angle (safety clamped)"""
        angle_deg = max(0, min(angle_deg, self.config["max_turn_deg"]))
        duration = angle_deg / 90.0  # Rough timing calibration
        
        self.pi.set_PWM_dutycycle(self.config["L_REV"], self.config["LEFT_SPEED"])
        self.pi.set_PWM_dutycycle(self.config["R_FWD"], self.config["RIGHT_SPEED"])
        time.sleep(duration)
        self.stop_motors()
        
        self.logger.info(f"Turned left {angle_deg}°")
    
    def turn_right(self, angle_deg: float):
        """Turn right by specified angle (safety clamped)"""
        angle_deg = max(0, min(angle_deg, self.config["max_turn_deg"]))
        duration = angle_deg / 90.0
        
        self.pi.set_PWM_dutycycle(self.config["L_FWD"], self.config["LEFT_SPEED"])
        self.pi.set_PWM_dutycycle(self.config["R_REV"], self.config["RIGHT_SPEED"])
        time.sleep(duration)
        self.stop_motors()
        
        self.logger.info(f"Turned right {angle_deg}°")
    
    def move_forward(self, duration_s: float):
        """Move forward for specified duration (safety clamped)"""
        duration_s = max(0.1, min(duration_s, self.config["max_move_s"]))
        
        self.pi.set_PWM_dutycycle(self.config["L_FWD"], self.config["LEFT_SPEED"])
        self.pi.set_PWM_dutycycle(self.config["R_FWD"], self.config["RIGHT_SPEED"])
        time.sleep(duration_s)
        self.stop_motors()
        
        self.logger.info(f"Moved forward {duration_s}s")
    
    def stop_motors(self):
        """Emergency stop all motors"""
        for pin in [self.config["L_FWD"], self.config["L_REV"],
                   self.config["R_FWD"], self.config["R_REV"]]:
            self.pi.set_PWM_dutycycle(pin, 0)

class RoboPiAgent:
    """
    Main Pi Agent following design doc architecture
    - Local AI for immediate responses
    - Tool API compliance
    - Observer state generation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize subsystems
        self.vision = VisionTools()
        self.motors = MotorControl()
        
        # Agent state
        self.current_behavior = "idle"
        self.target_person = None
        self.behavior_start_time = time.time()
        
        self.logger.info("RoboPi Agent initialized per design docs")
    
    def get_agent_state(self) -> Dict[str, Any]:
        """Get current agent state for reporting"""
        return {
            "timestamp": time.time(),
            "behavior": self.current_behavior,
            "target_person": self.target_person,
            "faces_now": self.vision.get_observer_state(),
            "behavior_duration": time.time() - self.behavior_start_time
        }
    
    def follow_person_behavior(self, duration_seconds: int = 60):
        """
        Basic person following behavior for milestone testing
        Implements simple reactive following as per design docs
        """
        self.logger.info(f"Starting person following for {duration_seconds} seconds")
        self.current_behavior = "following"
        self.behavior_start_time = time.time()
        
        end_time = time.time() + duration_seconds
        
        try:
            while time.time() < end_time:
                # Get current face situation
                faces = self.vision.detect_faces()
                state = self.get_agent_state()
                
                if faces:
                    face = faces[0]  # Primary face
                    self.logger.info(f"AGENT STATE: faces_now = \"{state['faces_now']}\"")
                    
                    # Simple reactive behavior
                    if face['position'] == 'LEFT':
                        self.motors.turn_left(15)
                        self.logger.info("Person is LEFT - turning left")
                    elif face['position'] == 'RIGHT':
                        self.motors.turn_right(15)
                        self.logger.info("Person is RIGHT - turning right")
                    elif face['position'] == 'CENTER':
                        if face['distance'] == 'FAR':
                            self.motors.move_forward(0.5)
                            self.logger.info("Person CENTERED - moving closer")
                        else:
                            self.logger.info(f"Person CENTERED - distance: {face['distance'].lower()}")
                    
                    # Behavior triggers based on confidence (design doc milestone)
                    if face['confidence'] >= 0.75:
                        self.logger.info(f"HIGH CONFIDENCE: Would speak('Hello {face['name']}')")
                    elif face['confidence'] < 0.45:
                        self.logger.info("LOW CONFIDENCE: Would speak('Hello! Can I help you?')")
                else:
                    self.logger.info("AGENT STATE: faces_now = \"None detected\"")
                    self.logger.info("No person detected - would search")
                
                time.sleep(0.5)  # ~2 Hz decision rate as per docs
                
        except KeyboardInterrupt:
            self.logger.info("Following behavior interrupted")
        finally:
            self.motors.stop_motors()
            self.current_behavior = "idle"
    
    def milestone_test(self, duration_seconds: int = 30):
        """
        First milestone test as specified in design docs:
        - MediaPipe detection at 640×480 ✓
        - MobileFaceNet embeddings and tiny gallery ✓  
        - Log {name,confidence,position,distance} to agent state ✓
        - Trigger simple behavior based on confidence ✓
        """
        self.logger.info("=== FIRST MILESTONE TEST ===")
        self.logger.info("MediaPipe + MobileFaceNet + tiny gallery")
        self.logger.info(f"Testing for {duration_seconds} seconds...")
        
        start_time = time.time()
        detection_count = 0
        
        while time.time() - start_time < duration_seconds:
            faces = self.vision.detect_faces()
            
            if faces:
                detection_count += 1
                for face in faces:
                    # Log format for Observer state per design docs
                    state_log = f"{face['name']} {face['position']} {face['distance']} ({face['confidence']:.2f})"
                    self.logger.info(f"AGENT STATE: faces_now = \"{state_log}\"")
                    
                    # Simple behavior triggers as per design docs
                    if face['confidence'] >= 0.75:
                        self.logger.info(f"HIGH CONFIDENCE: Would speak('Hello {face['name']}')")
                    elif face['confidence'] < 0.45:
                        self.logger.info("LOW CONFIDENCE: Would speak('Hello! Can I help you?')")
                    else:
                        self.logger.info(f"MEDIUM CONFIDENCE: Tracking {face['name']}")
            else:
                self.logger.info("AGENT STATE: faces_now = \"None detected\"")
            
            time.sleep(0.5)  # ~2 Hz as per docs
        
        self.logger.info("=== MILESTONE COMPLETE ===")
        self.logger.info(f"Detections in {duration_seconds}s: {detection_count}")
        self.logger.info("Ready for Executor/Planner integration")
        
        return {
            "duration": duration_seconds,
            "detections": detection_count,
            "fps": detection_count / duration_seconds,
            "status": "complete"
        }

def main():
    """Main entry point for Pi Agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RoboPi Agent (Design Doc)")
    parser.add_argument("--mode", choices=["milestone", "follow", "idle"], 
                       default="milestone", help="Operating mode")
    parser.add_argument("--duration", type=int, default=30, 
                       help="Duration for timed behaviors")
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        agent = RoboPiAgent()
        
        if args.mode == "milestone":
            agent.milestone_test(args.duration)
        elif args.mode == "follow":
            agent.follow_person_behavior(args.duration)
        elif args.mode == "idle":
            logging.info("Agent in idle mode - use agent.milestone_test() or agent.follow_person_behavior()")
            
    except KeyboardInterrupt:
        logging.info("Agent shutdown requested")
    except Exception as e:
        logging.error(f"Agent error: {e}")
        raise

if __name__ == "__main__":
    main()
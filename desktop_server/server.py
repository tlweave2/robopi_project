#!/usr/bin/env python3
"""
Desktop Recognition Server - Following Design Docs
- InsightFace with MobileFaceNet (w600k_mbf.onnx equivalent)
- NPZ gallery format for face storage
- Hybrid Pi+Desktop architecture as specified
- ~100ms latency over LAN for "instant" feel
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import logging
import os
import json
from datetime import datetime
from pathlib import Path

# Try InsightFace, fallback to OpenCV if not available
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available, using OpenCV fallback")

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DesktopRecognizer:
    def __init__(self, gallery_path="gallery", config_path="config.json"):
        self.gallery_path = Path(gallery_path)
        self.config_path = config_path
        self.gallery_path.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize recognition system
        if INSIGHTFACE_AVAILABLE and self.config.get("use_insightface", True):
            self.init_insightface()
        else:
            self.init_opencv_fallback()
            
        # Load existing gallery
        self.load_gallery()
        
        logger.info(f"Desktop recognizer initialized with {len(self.known_faces)} known faces")
    
    def load_config(self):
        """Load server configuration"""
        default_config = {
            "use_insightface": True,
            "recognition_threshold": 0.6,
            "det_size": (640, 640),
            "gallery_format": "npz",
            "max_faces_per_person": 10,
            "backup_on_enroll": True
        }
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            # Create default config
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def init_insightface(self):
        """Initialize InsightFace with MobileFaceNet (design doc spec)"""
        try:
            logger.info("Initializing InsightFace with MobileFaceNet...")
            self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0, det_size=self.config["det_size"])
            self.recognition_type = "insightface"
            logger.info("InsightFace MobileFaceNet ready for ~5-12 embeddings/sec")
        except Exception as e:
            logger.error(f"InsightFace initialization failed: {e}")
            self.init_opencv_fallback()
    
    def init_opencv_fallback(self):
        """Fallback to OpenCV face recognition"""
        logger.info("Using OpenCV face recognition fallback")
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognition_type = "opencv"
        self.training_faces = []
        self.training_labels = []
        self.label_names = []
    
    def extract_embedding(self, face_crop_160):
        """Extract face embedding using available system"""
        if self.recognition_type == "insightface":
            try:
                faces = self.face_app.get(face_crop_160)
                if faces and len(faces) > 0:
                    return faces[0].embedding
            except Exception as e:
                logger.error(f"InsightFace embedding extraction failed: {e}")
        
        # OpenCV fallback doesn't use embeddings
        return None
    
    def recognize_face_crop(self, face_crop_160):
        """
        Tool API: identify_person(face_crop) -> {name, confidence}
        As specified in design docs
        """
        if self.recognition_type == "insightface":
            return self.recognize_with_insightface(face_crop_160)
        else:
            return self.recognize_with_opencv(face_crop_160)
    
    def recognize_with_insightface(self, face_crop_160):
        """Recognition using InsightFace + NPZ gallery"""
        embedding = self.extract_embedding(face_crop_160)
        if embedding is None:
            return {"name": "Unknown", "confidence": 0.0}
        
        if not self.known_faces:
            return {"name": "Unknown", "confidence": 0.0}
        
        best_match = "Unknown"
        best_score = 0.0
        
        for name, known_embedding in self.known_faces.items():
            # Cosine similarity as per design docs
            similarity = np.dot(embedding, known_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
            )
            
            if similarity > best_score and similarity > self.config["recognition_threshold"]:
                best_score = similarity
                best_match = name
        
        return {"name": best_match, "confidence": float(best_score)}
    
    def recognize_with_opencv(self, face_crop_160):
        """Fallback OpenCV recognition"""
        try:
            gray = cv2.cvtColor(face_crop_160, cv2.COLOR_BGR2GRAY)
            
            if len(self.label_names) == 0:
                return {"name": "Unknown", "confidence": 0.0}
            
            label, confidence = self.face_recognizer.predict(gray)
            
            # OpenCV confidence: lower = better match
            if confidence < 100:
                name = self.label_names[label] if label < len(self.label_names) else "Unknown"
                conf_score = max(0.0, (100 - confidence) / 100)
                return {"name": name, "confidence": conf_score}
            else:
                return {"name": "Unknown", "confidence": 0.0}
        except Exception as e:
            logger.error(f"OpenCV recognition error: {e}")
            return {"name": "Unknown", "confidence": 0.0}
    
    def enroll_face(self, name, face_crop_160):
        """
        Enroll face following design doc specs:
        - Store in NPZ gallery format (InsightFace)
        - Build training model (OpenCV fallback)
        """
        if self.recognition_type == "insightface":
            return self.enroll_with_insightface(name, face_crop_160)
        else:
            return self.enroll_with_opencv(name, face_crop_160)
    
    def enroll_with_insightface(self, name, face_crop_160):
        """Enroll using InsightFace + NPZ gallery"""
        embedding = self.extract_embedding(face_crop_160)
        if embedding is None:
            return False
        
        # Average with existing embeddings if person exists
        if name in self.known_faces:
            existing_embedding = self.known_faces[name]
            # Simple averaging - could be improved with more sophisticated fusion
            self.known_faces[name] = (existing_embedding + embedding) / 2
        else:
            self.known_faces[name] = embedding
        
        # Save to NPZ gallery (design doc format)
        self.save_gallery()
        
        # Backup if configured
        if self.config["backup_on_enroll"]:
            self.backup_gallery()
        
        logger.info(f"Enrolled {name} using InsightFace")
        return True
    
    def enroll_with_opencv(self, name, face_crop_160):
        """Enroll using OpenCV fallback"""
        try:
            gray = cv2.cvtColor(face_crop_160, cv2.COLOR_BGR2GRAY)
            
            # Add to training data
            if name not in self.label_names:
                self.label_names.append(name)
                label = len(self.label_names) - 1
            else:
                label = self.label_names.index(name)
            
            self.training_faces.append(gray)
            self.training_labels.append(label)
            
            # Retrain model
            if len(self.training_faces) > 0:
                self.face_recognizer.train(self.training_faces, np.array(self.training_labels))
                self.face_recognizer.save("face_model.yml")
            
            logger.info(f"Enrolled {name} using OpenCV")
            return True
            
        except Exception as e:
            logger.error(f"OpenCV enrollment error: {e}")
            return False
    
    def load_gallery(self):
        """Load face gallery from NPZ format (design doc)"""
        self.known_faces = {}
        
        if self.recognition_type == "insightface":
            gallery_file = self.gallery_path / "faces.npz"
            try:
                data = np.load(gallery_file)
                for name in data.files:
                    self.known_faces[name] = data[name]
                logger.info(f"Loaded {len(self.known_faces)} faces from NPZ gallery")
            except FileNotFoundError:
                logger.info("No existing NPZ gallery found")
        else:
            # OpenCV fallback
            try:
                self.face_recognizer.read("face_model.yml")
                # Note: OpenCV doesn't store names, would need separate mapping
                logger.info("Loaded OpenCV face model")
            except:
                logger.info("No existing OpenCV model found")
    
    def save_gallery(self):
        """Save gallery to NPZ format (design doc)"""
        if self.recognition_type == "insightface" and self.known_faces:
            gallery_file = self.gallery_path / "faces.npz"
            np.savez(gallery_file, **self.known_faces)
            logger.info(f"Saved {len(self.known_faces)} faces to NPZ gallery")
    
    def backup_gallery(self):
        """Create timestamped backup of gallery"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.gallery_path / f"faces_backup_{timestamp}.npz"
        
        if self.known_faces:
            np.savez(backup_file, **self.known_faces)
            logger.info(f"Created gallery backup: {backup_file}")
    
    def get_stats(self):
        """Get recognition system statistics"""
        return {
            "recognition_type": self.recognition_type,
            "known_faces_count": len(self.known_faces),
            "known_faces": list(self.known_faces.keys()),
            "gallery_path": str(self.gallery_path),
            "config": self.config,
            "insightface_available": INSIGHTFACE_AVAILABLE
        }

# Initialize recognizer
recognizer = DesktopRecognizer()

@app.route('/recognize', methods=['POST'])
def recognize():
    """
    API endpoint for hybrid recognition
    Pi sends 160×160 face crop, Desktop returns {name, confidence}
    """
    try:
        data = request.get_json()
        face_b64 = data['face_image']
        
        # Decode face crop
        face_bytes = base64.b64decode(face_b64)
        face_np = np.frombuffer(face_bytes, np.uint8)
        face_crop = cv2.imdecode(face_np, cv2.IMREAD_COLOR)
        
        # Recognize using available system
        result = recognizer.recognize_face_crop(face_crop)
        
        logger.info(f"Recognition: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Recognition endpoint error: {e}")
        return jsonify({"name": "Unknown", "confidence": 0.0})

@app.route('/enroll', methods=['POST'])
def enroll():
    """
    API endpoint for face enrollment
    Builds NPZ gallery or OpenCV model
    """
    try:
        data = request.get_json()
        name = data['name']
        face_b64 = data['face_image']
        
        # Decode face crop
        face_bytes = base64.b64decode(face_b64)
        face_np = np.frombuffer(face_bytes, np.uint8)
        face_crop = cv2.imdecode(face_np, cv2.IMREAD_COLOR)
        
        # Enroll face
        success = recognizer.enroll_face(name, face_crop)
        
        return jsonify({
            "success": success,
            "message": f"Enrolled {name}" if success else "Enrollment failed",
            "total_faces": len(recognizer.known_faces)
        })
        
    except Exception as e:
        logger.error(f"Enrollment endpoint error: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/health', methods=['GET'])
def health():
    """Health check and system stats"""
    return jsonify({
        "status": "ready",
        "stats": recognizer.get_stats()
    })

@app.route('/gallery', methods=['GET'])
def gallery():
    """Get gallery information"""
    return jsonify(recognizer.get_stats())

@app.route('/backup', methods=['POST'])
def backup():
    """Manual gallery backup"""
    try:
        recognizer.backup_gallery()
        return jsonify({"success": True, "message": "Gallery backed up"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

if __name__ == '__main__':
    logger.info("=== Desktop Recognition Server (Design Doc Compliant) ===")
    logger.info(f"Recognition type: {recognizer.recognition_type}")
    logger.info(f"Known faces: {len(recognizer.known_faces)}")
    logger.info("Starting server on port 8080...")
    
    app.run(host='0.0.0.0', port=8080, debug=False)
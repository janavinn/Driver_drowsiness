import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import winsound
import os
import threading
import subprocess

# ---------------- CONFIG ----------------
# Files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'efficientdet_lite0.tflite')

# Thresholds
EYE_AR_THRESH = 0.15      
EYE_AR_CONSEC_FRAMES = 15 
YAWN_THRESH = 0.6         
PHONE_CONSEC_FRAMES = 30  # ~1 second at 30fps (since we check every frame or so)

# ---------------- INITIALIZATION ----------------
print("[INFO] Initializing Mediapipe...")
# 1. Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Object Detector (Phone Search)
HAS_DETECTOR = False
try:
    if os.path.exists(MODEL_PATH):
        # Read file as binary to avoid Windows path issues in C++ layer
        with open(MODEL_PATH, 'rb') as f:
            model_buffer = f.read()
            
        base_options = python.BaseOptions(model_asset_buffer=model_buffer)
        options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.3)
        detector = vision.ObjectDetector.create_from_options(options)
        HAS_DETECTOR = True
        print("[INFO] Object Detector loaded.")
    else:
        print(f"[WARNING] Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Logic error in detector init: {e}")

# Landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Mouth
MOUTH_TOP = 13; MOUTH_BOTTOM = 14; MOUTH_LEFT = 61; MOUTH_RIGHT = 291

# ---------------- STATE VARIABLES ----------------
COUNTER = 0           
ALARM_ON = False      
last_alarm_time = 0
frame_count = 0
PHONE_COUNTER = 0
PHONE_DETECTED = False
PHONE_BOX = None

# ---------------- METRIC FUNCTIONS ----------------
def average_ear(landmarks, indices, width, height):
    # Eye Aspect Ratio
    pts = []
    for idx in indices:
        pt = landmarks[idx]
        pts.append(np.array([pt.x * width, pt.y * height]))
    
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    
    return (A + B) / (2.0 * C) if C > 1e-6 else 0

def mouth_aspect_ratio(landmarks, width, height):
    top = landmarks[MOUTH_TOP]; bottom = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]; right = landmarks[MOUTH_RIGHT]
    
    vertical = np.linalg.norm(np.array([top.x*w, top.y*h]) - np.array([bottom.x*w, bottom.y*h]))
    horizontal = np.linalg.norm(np.array([left.x*w, left.y*h]) - np.array([right.x*w, right.y*h]))
    
    return vertical / horizontal if horizontal > 1e-6 else 0

def play_windows_beep(duration=700, freq=850):
    try:
        winsound.Beep(int(freq), int(duration))
    except RuntimeError:
        pass

def speak_alert(text):
    """Speaks the text using PowerShell TTS in a background thread."""
    def _speak():
        try:
            # -WindowStyle Hidden keeps it relatively quiet, though PS might still flash briefly
            cmd = f'PowerShell -Command "Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\');"'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    
    threading.Thread(target=_speak, daemon=True).start()

# ---------------- MAIN LOOP ----------------
def live_face_detection():
    global COUNTER, ALARM_ON, last_alarm_time, frame_count, w, h, PHONE_COUNTER, PHONE_DETECTED, PHONE_BOX
    
    print("[INFO] Starting video stream...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not access webcam.")
        return

    time.sleep(1) 

    # Feature Variables
    start_time = time.time()
    prev_frame_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # --- FEATURE 1 & 2: FPS and Timer ---
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time
            
            elapsed_seconds = int(new_frame_time - start_time)
            timer_str = f"Time: {elapsed_seconds // 3600:02}:{(elapsed_seconds % 3600) // 60:02}:{elapsed_seconds % 60:02}"
            fps_str = f"FPS: {int(fps)}"

            clean_frame = frame.copy()
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- FEATURE 3: Low Light Detection ---
            # Convert to gray to check brightness
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray_frame)
            
            # --- 1. Object Detection (Freq: Every 3 frames) ---
            if HAS_DETECTOR and frame_count % 3 == 0:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = detector.detect(mp_image)
                
                PHONE_DETECTED = False
                PHONE_BOX = None
                for detection in detection_result.detections:
                    if detection.categories[0].category_name == 'cell phone':
                        PHONE_DETECTED = True
                        PHONE_BOX = detection.bounding_box
                        break
            
            # --- 2. Face Mesh ---
            results = face_mesh.process(rgb_frame)
            
            status_text = "System Active"
            state = "Normal"
            color = (0, 255, 0)
            
            # Check Phone Detection State
            if PHONE_DETECTED:
                PHONE_COUNTER += 1
            else:
                PHONE_COUNTER = 0

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lm = face_landmarks.landmark
                    
                    # Metrics
                    left_ear = average_ear(lm, LEFT_EYE, w, h)
                    right_ear = average_ear(lm, RIGHT_EYE, w, h)
                    avg_ear = (left_ear + right_ear) / 2.0
                    mar = mouth_aspect_ratio(lm, w, h)

                    # --- PRIORITY LOGIC ---
                    
                    # 1. Phone Alert (Top Priority)
                    if PHONE_COUNTER > 10:
                        status_text = "PHONE USAGE DETECTED"
                        color = (0, 0, 255) # Red for Alert
                        if time.time() - last_alarm_time > 1.5:
                            play_windows_beep(600, 1000)
                            speak_alert("Phone usage detected")
                            last_alarm_time = time.time()
                    
                    # 2. Drowsiness
                    elif avg_ear < EYE_AR_THRESH:
                        COUNTER += 1
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            status_text = "DROWSINESS DETECTED"
                            color = (0, 0, 255)
                            if time.time() - last_alarm_time > 1.0:
                                play_windows_beep(800, 1000)
                                speak_alert("Drowsiness detected")
                                last_alarm_time = time.time()
                    
                    # 3. Yawning
                    elif mar > YAWN_THRESH:
                        status_text = "YAWNING DETECTED"
                        color = (0, 255, 255) # Yellow
                        COUNTER = 0
                        if time.time() - last_alarm_time > 2.0:
                            play_windows_beep(400, 700)
                            speak_alert("Yawning detected")
                            last_alarm_time = time.time()
                    else:
                        COUNTER = 0

                    # Drawing Landmarks
                    for idx in LEFT_EYE + RIGHT_EYE:
                        pt = lm[idx]
                        cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 1, (255, 0, 0), -1)
                    
                    # Draw Face Bounding Box (Green)
                    x_list = [l.x for l in lm]
                    y_list = [l.y for l in lm]
                    x_min, x_max = int(min(x_list) * w), int(max(x_list) * w)
                    y_min, y_max = int(min(y_list) * h), int(max(y_list) * h)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Draw Phone Box (Green as requested)
                    if PHONE_DETECTED and PHONE_BOX:
                         cv2.rectangle(frame, (PHONE_BOX.origin_x, PHONE_BOX.origin_y), 
                                      (PHONE_BOX.origin_x + PHONE_BOX.width, PHONE_BOX.origin_y + PHONE_BOX.height), (0, 255, 0), 3)

                    # Debug Text (EAR/MAR) - Blue Color (255, 0, 0)
                    cv2.putText(frame, f"EAR: {avg_ear:.2f} | MAR: {mar:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Draw Status
            cv2.putText(frame, status_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
            # Draw Low Light Warning
            if avg_brightness < 80: # Threshold for dark room
                cv2.putText(frame, "POOR LIGHTING", (w - 250, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Stack images
            # Add Headers (Restored UI)
            cv2.putText(clean_frame, "Original Video", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(clean_frame, f"{timer_str} | {fps_str}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.putText(frame, "Detection Output", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            composite = np.hstack((clean_frame, frame))
            cv2.imshow("Driver Monitoring System Pro", composite)
            
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
if __name__ == '__main__':
    live_face_detection()

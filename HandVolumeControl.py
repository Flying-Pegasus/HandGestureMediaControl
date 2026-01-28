import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import time
from math import hypot as hypo
import numpy as np
import urllib.request
import os

# ======================== CONFIGURATION ======================== #
wCam, hCam = 640, 480          # Camera resolution
screenW, screenH = pyautogui.size()  # Get actual screen resolution

# Mouse control parameters
SMOOTHING = 2                   # Higher = smoother but slower cursor (1-10)
FRAME_REDUCTION = 100           # Border margin for hand tracking area
PINCH_THRESHOLD = 40            # Distance threshold for index-thumb pinch
CLICK_THRESHOLD = 50            # Distance threshold for middle finger click
CLICK_COOLDOWN = 0.3            # Seconds between clicks to prevent double-clicks

# Disable pyautogui fail-safe for edge movements (optional)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0             # Remove delay between pyautogui commands
# ============================================================== #

# Download hand landmarker model if not exists
MODEL_PATH = os.path.expanduser("~/.mediapipe/hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Downloading hand landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded!")

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize MediaPipe Hand Landmarker with new Tasks API
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# State variables
mouse_active = False
prev_x, prev_y = 0, 0           # Previous cursor position for smoothing
last_click_time = 0             # For click cooldown
click_ready = True              # Prevents continuous clicking

# FPS calculation
pTime = 0

def get_landmark_pos(lmList, idx):
    """Get x, y coordinates for a specific landmark"""
    return lmList[idx][1], lmList[idx][2]

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return hypo(x2 - x1, y2 - y1)

def map_coordinates(x, y, frame_w, frame_h):
    """Map hand coordinates to screen coordinates with boundary handling"""
    # Define the active region within the frame (with margins)
    x_min, x_max = FRAME_REDUCTION, frame_w - FRAME_REDUCTION
    y_min, y_max = FRAME_REDUCTION, frame_h - FRAME_REDUCTION
    
    # Clamp coordinates to active region
    x = np.clip(x, x_min, x_max)
    y = np.clip(y, y_min, y_max)
    
    # Map to screen coordinates
    screen_x = np.interp(x, (x_min, x_max), (0, screenW))
    screen_y = np.interp(y, (y_min, y_max), (0, screenH))
    
    return screen_x, screen_y

def smooth_cursor(curr_x, curr_y, prev_x, prev_y, smoothing):
    """Apply smoothing to cursor movement for fluid motion"""
    smooth_x = prev_x + (curr_x - prev_x) / smoothing
    smooth_y = prev_y + (curr_y - prev_y) / smoothing
    return smooth_x, smooth_y

print("=" * 50)
print("HAND MOUSE CONTROL")
print("=" * 50)
print("GESTURES:")
print("  • Pinch Index + Thumb  → Activate mouse mode")
print("  • Move hand            → Move cursor")
print("  • Add Middle finger    → Click")
print("  • Release pinch        → Deactivate mouse")
print("  • Press ESC            → Exit program")
print("=" * 50)

def draw_hand_landmarks(img, landmarks, w, h):
    """Draw hand landmarks and connections on the image"""
    # Define hand connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm
    ]
    
    # Draw connections
    for start, end in connections:
        x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
        x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw landmarks
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), cv2.FILLED)

while True:
    success, img = cap.read()
    if not success:
        continue
        
    img = cv2.flip(img, 1)  # Mirror image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    
    # Detect hands
    results = detector.detect(mp_image)
    
    h, w, c = img.shape
    lmList = []
    
    # Draw active region boundary
    cv2.rectangle(img, (FRAME_REDUCTION, FRAME_REDUCTION), 
                  (w - FRAME_REDUCTION, h - FRAME_REDUCTION), 
                  (255, 0, 255), 2)
    
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Extract all landmarks
            for id, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            if len(lmList) >= 21:  # Ensure all landmarks detected
                # Get key landmark positions
                x4, y4 = get_landmark_pos(lmList, 4)    # Thumb tip
                x8, y8 = get_landmark_pos(lmList, 8)    # Index finger tip
                x12, y12 = get_landmark_pos(lmList, 12) # Middle finger tip
                x0, y0 = get_landmark_pos(lmList, 0)    # Wrist
                x9, y9 = get_landmark_pos(lmList, 9)    # Middle finger base
                
                # Calculate palm length for relative measurements
                palm_length = calculate_distance(x0, y0, x9, y9)
                
                # Calculate distances for gestures
                pinch_distance = calculate_distance(x4, y4, x8, y8)  # Index-Thumb
                click_distance = calculate_distance(x12, y12, x8, y8)  # Middle-Index
                
                # Dynamic threshold based on hand size
                pinch_thresh = min(PINCH_THRESHOLD, palm_length * 0.3)
                click_thresh = min(CLICK_THRESHOLD, palm_length * 0.35)
                
                # ============ MOUSE ACTIVATION (Index + Thumb pinch) ============
                if pinch_distance < pinch_thresh:
                    mouse_active = True
                    
                    # Use index finger position for cursor control
                    cursor_x, cursor_y = x8, y8
                    
                    # Map to screen coordinates
                    screen_x, screen_y = map_coordinates(cursor_x, cursor_y, w, h)
                    
                    # Apply smoothing
                    smooth_x, smooth_y = smooth_cursor(screen_x, screen_y, 
                                                       prev_x, prev_y, SMOOTHING)
                    
                    # Move cursor
                    pyautogui.moveTo(int(smooth_x), int(smooth_y))
                    
                    # Update previous position
                    prev_x, prev_y = smooth_x, smooth_y
                    
                    # Visual feedback - pinch point
                    cx_pinch = (x4 + x8) // 2
                    cy_pinch = (y4 + y8) // 2
                    cv2.circle(img, (cx_pinch, cy_pinch), 15, (0, 255, 0), cv2.FILLED)
                    cv2.circle(img, (x8, y8), 10, (0, 255, 255), cv2.FILLED)
                    
                    # ============ CLICK DETECTION (Middle finger tap) ============
                    current_time = time.time()
                    
                    if click_distance < click_thresh:
                        if click_ready and (current_time - last_click_time) > CLICK_COOLDOWN:
                            pyautogui.click()
                            last_click_time = current_time
                            click_ready = False
                            # Visual feedback for click
                            cv2.circle(img, (cx_pinch, cy_pinch), 25, (0, 0, 255), cv2.FILLED)
                            print("Click!")
                    else:
                        click_ready = True  # Reset click when middle finger is released
                        
                else:
                    # Deactivate mouse when pinch is released
                    if mouse_active:
                        mouse_active = False
                        print("Mouse deactivated")
                
                # Draw landmarks on thumb and fingers
                cv2.circle(img, (x4, y4), 8, (255, 0, 0), cv2.FILLED)   # Thumb - Blue
                cv2.circle(img, (x8, y8), 8, (0, 255, 0), cv2.FILLED)   # Index - Green
                cv2.circle(img, (x12, y12), 8, (0, 0, 255), cv2.FILLED) # Middle - Red
                
                # Draw connection lines
                cv2.line(img, (x4, y4), (x8, y8), (255, 255, 0), 2)
                
            # Draw hand skeleton using custom function
            draw_hand_landmarks(img, hand_landmarks, w, h)
    
    # Display status
    status_color = (0, 255, 0) if mouse_active else (0, 0, 255)
    status_text = "MOUSE: ACTIVE" if mouse_active else "MOUSE: INACTIVE"
    cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Instructions on screen
    cv2.putText(img, "Pinch to activate | Middle tap to click | ESC to exit", 
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow("Hand Mouse Control", img)
    
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended.")

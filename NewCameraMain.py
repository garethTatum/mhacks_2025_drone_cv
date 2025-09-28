import cv2
import numpy as np
# import Jetson.GPIO as GPIO  # <<< DISABLED: This only works on a Jetson
import time

# --- 1. HARDWARE & PUMP SETUP (SIMULATED for PC/Mac) ---
# PUMP_PIN = 12 
# GPIO.setmode(GPIO.BOARD) 
# GPIO.setup(PUMP_PIN, GPIO.OUT, initial=GPIO.LOW)

# --- Camera Setup (same as before) ---
cap = cv2.VideoCapture(0)
KNOWN_REAL_DIMENSION_CM = 30.48
FOCAL_LENGTH_PIXELS = 2571
SAFE_CIRCLE_RADIUS_PX = 110
TOLERANCE_PX = 10
ret, temp = cap.read()
if not ret:
    raise RuntimeError("Cannot read from camera")
H, W = temp.shape[:2]
center_img_x = W // 2
center_img_y = H // 2
center_img = (center_img_x, center_img_y)

# --- Variables for non-blocking pump timer ---
pump_is_on = False
pump_start_time = 0
PUMP_DURATION_S = 1 # Pump will run for 2 seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        output = frame.copy()
        
        # --- Image Processing (same as before) ---
        blurred = cv2.GaussianBlur(frame, (9, 9), 2)
        hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 70, 70])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        lower_red2 = np.array([160, 70, 70])
        upper_red2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        color_mask = mask1 + mask2
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- Contour Processing (same as before) ---
        found_valid_cup = False
        at_least_one_cup_is_correct = False
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if area > 500 and perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
                if 0.6 < circularity < 1.2:
                    found_valid_cup = True
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    center = (int(x), int(y))
                    dist_from_center_px = np.sqrt((center[0] - center_img_x)**2 + (center[1] - center_img_y)**2)
                    if not (dist_from_center_px > SAFE_CIRCLE_RADIUS_PX + TOLERANCE_PX):
                        at_least_one_cup_is_correct = True
                    cv2.circle(output, center, int(radius), (0, 255, 0), 4)

        # --- Summary Status Drawing (same as before) ---
        if at_least_one_cup_is_correct:
            status_text = "Status: CORRECT DISTANCE"
            circle_color = (0, 255, 0)
        elif found_valid_cup:
            status_text = "Status: TARGETS TOO FAR"
            circle_color = (0, 255, 255)
        else:
            status_text = "Status: No Target"
            circle_color = (0, 0, 255)
        cv2.circle(output, center_img, SAFE_CIRCLE_RADIUS_PX, circle_color, 2)
        cv2.putText(output, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, circle_color, 2)

        # --- 4. PUMP CONTROL LOGIC (SIMULATED) ---
        if at_least_one_cup_is_correct and not pump_is_on:
            print(">>> PUMP ON (SIMULATED) <<<") # <<< CHANGED: Now a print statement
            # GPIO.output(PUMP_PIN, GPIO.HIGH) # Disabled
            pump_is_on = True
            pump_start_time = time.time()

        if pump_is_on and (time.time() - pump_start_time > PUMP_DURATION_S):
            print(">>> PUMP OFF (SIMULATED) <<<") # <<< CHANGED: Now a print statement
            # GPIO.output(PUMP_PIN, GPIO.LOW) # Disabled
            pump_is_on = False

        cv2.imshow("Drone View", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Exiting...")
    # GPIO.cleanup() # <<< DISABLED: No GPIO to clean up
    cap.release()
    cv2.destroyAllWindows()
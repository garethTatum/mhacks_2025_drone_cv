import cv2
import numpy as np

# --- 1. SETUP ---
cap = cv2.VideoCapture(0)

KNOWN_REAL_DIMENSION_CM = 30.48 # 1 foot = 30.48 cm
FOCAL_LENGTH_PIXELS = 2571

# Safe Circle Setup
SAFE_CIRCLE_RADIUS_PX = 150
TOLERANCE_PX = 20

# Get frame dimensions and calculate center
ret, temp = cap.read()
if not ret:
    raise RuntimeError("Cannot read from camera")
H, W = temp.shape[:2]
center_img_x = W // 2
center_img_y = H // 2
center_img = (center_img_x, center_img_y)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    output = frame.copy()
    
    # Initialize status and color for this frame
    status_text = "Status: No Target"
    coord_text = "Coords: N/A" # Default coordinate text
    circle_color = (0, 0, 255) # Default to RED

    # Pre-processing
    blurred = cv2.GaussianBlur(frame, (9, 9), 2)
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Red color mask
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    lower_red2 = np.array([160, 70, 70])
    upper_red2 = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    color_mask = mask1 + mask2

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if area > 500 and perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            if 0.6 < circularity < 1.2:
                (x, y), radius = cv2.minEnclosingCircle(c)
                center = (int(x), int(y))
                radius = int(radius)
                
                # --- ADDED BACK: Real-world distance calculations ---
                pixel_diameter = radius * 2
                if pixel_diameter > 0:
                    # Calculate distance away from camera (Z-axis)
                    distance_cm = (KNOWN_REAL_DIMENSION_CM * FOCAL_LENGTH_PIXELS) / pixel_diameter
                    # Calculate real-world X and Y offsets
                    real_x_cm = ((center[0] - center_img_x) * distance_cm) / FOCAL_LENGTH_PIXELS
                    real_y_cm = ((center[1] - center_img_y) * distance_cm) / FOCAL_LENGTH_PIXELS
                    # Update coordinate text
                    coord_text = f"Coords: X:{real_x_cm:.1f} Y:{real_y_cm:.1f} Z:{distance_cm:.1f} (cm)"
                # ---------------------------------------------------

                # Check position relative to safe circle
                dist_from_center_px = np.sqrt((center[0] - center_img_x)**2 + (center[1] - center_img_y)**2)
                if dist_from_center_px > SAFE_CIRCLE_RADIUS_PX + TOLERANCE_PX:
                    status_text = "Status: TOO FAR"
                    circle_color = (0, 255, 255) # YELLOW
                else:
                    status_text = "Status: CORRECT DISTANCE"
                    circle_color = (0, 255, 0) # GREEN

                # Draw the detected circle and its center
                cv2.circle(output, center, radius, (0, 255, 0), 4)
                cv2.circle(output, center, 2, (0, 0, 255), -1)

    # Draw the safe circle and status text on every frame
    cv2.circle(output, center_img, SAFE_CIRCLE_RADIUS_PX, circle_color, 2)
    cv2.putText(output, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, circle_color, 2)
    
    # --- ADDED: Display for real-world coordinates ---
    cv2.putText(output, coord_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow("Drone View", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
from bottom import BottomCamera
from FrontCamera import FrontCamera
import cv2

# --- CONFIGURATION ---
### NTOE TO SELF: WE NEED TO ADJUST THE CAMERA INDEX AFTERWARDS
BOTTOM_CAMERA_INDEX = 0
FRONT_CAMERA_INDEX = 1

# Yolo model weight path
MODEL_PATH = "detect/rsc_detection3/weights/best.pt" 

# Use the real diameter of your cup for accurate distances
KNOWN_OBJECT_DIAMETER = 9.5 
OBJECT_REAL_HEIGHT_M = 0.114 
CAMERA_FOV_DEG = 60 
FRONT_CAMERA_DIST_THRESHOLD_M = 0.4 # 40 cm

# Or, use the 1-foot value as previously requested
# KNOWN_OBJECT_DIAMETER = 30.48 
FOCAL_LENGTH = 2571

def main():
    bottomTracker = None
    frontTracker = None
    try:
        # 1. Initialize the tracker with your settings
        bottomTracker = BottomCamera(BOTTOM_CAMERA_INDEX, KNOWN_OBJECT_DIAMETER, FOCAL_LENGTH)
        frontTracker = FrontCamera(FRONT_CAMERA_INDEX, MODEL_PATH)

        # stage 1: camera 1 and camera 2
        current_stage = 1
        while True:
            # 2. Read a frame from the camera
            ret_bottom, frame_bottom = bottomTracker.cap.read()
            ret_front, frame_front = frontTracker.cap.read()

            if not ret_bottom or not ret_front:
                print("Failed to capture frame. Exiting.")
                break

            # 3. Process the frame to get all tracking data
            if current_stage == 1:
                data_front = front_tracker.process_frame(frame_front, OBJECT_REAL_HEIGHT_M, CAMERA_FOV_DEG)
                front_tracker.draw_visuals(frame_front, data_front)

                if data_front["target_found"]:
                    dist = data_front['distance_m']
                    angle = data_front['angle_deg']
                    print(f"[Stage 1] Approaching: Dist: {dist:.2f}m, Angle: {angle:.1f}deg")
                    # SEND INFO TO PIXHAWLK
                    # Check for transition condition
                    if dist < FRONT_CAMERA_DIST_THRESHOLD_M:
                        print(">>> Target is close. Transitioning to Stage 2 (Centering)...")
                        current_stage = 2
                else:
                    print("[Stage 1] Searching for target...")
            
            elif current_stage == 2:
                # Process bottom camera for precise centering
                data_bottom, _, _ = bottom_tracker.process_frame(frame_bottom)
                bottom_tracker.draw_visuals(frame_bottom, data_bottom)
                
                if data_bottom["target_found"]:
                    # Calculate the distance in pixels the drone needs to move
                    dist_to_move_px = data_bottom["dist_from_center_px"]
                    
                    # Calculate the vector from the image center to the target's center
                    dx = data_bottom["center_px"][0] - bottom_tracker.center_img_x
                    dy = data_bottom["center_px"][1] - bottom_tracker.center_img_y
                    
                    # Calculate the angle of that vector for the drone's direction
                    angle_to_move_deg = np.degrees(np.arctan2(dy, dx))

                    print(f"[Stage 2] Command: Move {dist_to_move_px:.1f}px towards {angle_to_move_deg:.1f} degrees")

                    # Transition condition
                    if data_bottom["status"] == "Status: CORRECT DISTANCE":
                        print(">>> Centered over target. Transitioning to Stage 3 (Dispense)...")
                        current_stage = 3
                else:
                    print("[Stage 2] Searching for landing target...")

            elif current_stage == 3:
                print("[Stage 3] DISPENSING... Mission Complete.")
                time.sleep(2) # Simulate dispensing action
                break # Exit the main loop
            
            if current_stage != 2: # Avoid reprocessing bottom camera in stage 2
                data_bottom, _, _ = bottom_tracker.process_frame(frame_bottom)
                bottom_tracker.draw_visuals(frame_bottom, data_bottom)
            
            if current_stage != 1: # Avoid reprocessing front camera in stage 1
                data_front = front_tracker.process_frame(frame_front, OBJECT_REAL_HEIGHT_M, CAMERA_FOV_DEG)
                front_tracker.draw_visuals(frame_front, data_front)

            cv2.imshow("Bottom Camera View", frame_bottom)
            cv2.imshow("Front Camera View (YOLO)", frame_front)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 7. Release resources cleanly
        if bottomTracker:
            bottomTracker.release()

if __name__ == "__main__":
    main()

from bottom import BottomCamera
import cv2

# --- CONFIGURATION ---
### NTOE TO SELF: WE NEED TO ADJUST THE CAMERA INDEX AFTERWARDS
BOTTOM_CAMERA_INDEX = 0
# Use the real diameter of your cup for accurate distances
KNOWN_OBJECT_DIAMETER = 9.5 
# Or, use the 1-foot value as previously requested
# KNOWN_OBJECT_DIAMETER = 30.48 
FOCAL_LENGTH = 2571

def main():
    bottomTracker = None
    try:
        # 1. Initialize the tracker with your settings
        bottomTracker = BottomCamera(BOTTOM_CAMERA_INDEX, KNOWN_OBJECT_DIAMETER, FOCAL_LENGTH)

        while True:
            # 2. Read a frame from the camera
            ret, frame = bottomTracker.cap.read()
            if not ret:
                print("Failed to capture frame. Exiting.")
                break

            # 3. Process the frame to get all tracking data
            data, output_frame, color_mask = bottomTracker.process_frame(frame)

            # 4. Use the data for your drone's control logic
            #    For this example, we just print the results to the console.
            if data["target_found"]:
                print(f'{data["status"]} | Coords: {data["coords_cm"]}')
            else:
                print(data["status"])
                
            # 5. Draw the visual feedback onto the frame
            bottomTracker.draw_visuals(output_frame, data)

            # 6. Show the output frames
            cv2.imshow("Drone View", output_frame)
            # cv2.imshow("Color Mask", color_mask)

            # Check for 'q' key to quit
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


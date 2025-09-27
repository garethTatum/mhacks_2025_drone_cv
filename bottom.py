import cv2
import numpy as np

class BottomCamera:
    """
    A class to detect a circular object of a specific color, calculate its
    real-world 3D coordinates, and determine its position relative to a
    pre-defined "safe circle" for drone stabilization.
    """

    def __init__(self, camera_index, known_object_diameter_cm, focal_length_pixels):
        """
        Initializes the bottom m Camera.

        Args:
            camera_index (int): The index of the camera to use (e.g., 0 for default).
            known_object_diameter_cm (float): The real-world diameter of the object.
            focal_length_pixels (int): The camera's focal length in pixels.
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {camera_index}")

        self.known_dimension_cm = known_object_diameter_cm
        self.focal_length_px = focal_length_pixels

        # Get frame dimensions and calculate center
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read initial frame from camera")
        self.H, self.W = frame.shape[:2]
        self.center_img_x = self.W // 2
        self.center_img_y = self.H // 2
        self.center_img = (self.center_img_x, self.center_img_y)

        # --- Hardcoded Settings ---
        # Safe circle parameters
        self.safe_radius_px = 125
        self.tolerance_px = 20

        # Target color is fixed to red
        self.hsv_ranges = [
            (np.array([0, 70, 70]), np.array([10, 255, 255])),
            (np.array([160, 70, 70]), np.array([179, 255, 255]))
        ]

    def process_frame(self, frame):
        """
        Processes a single frame to find the target and calculate data.

        Args:
            frame (np.array): The camera frame to process.

        Returns:
            dict: A dictionary containing tracking data.
            np.array: The original frame for drawing visuals.
        """
        output_frame = frame.copy()
        
        # --- Pre-processing ---
        blurred = cv2.GaussianBlur(frame, (9, 9), 2)
        hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # --- Color Masking ---
        color_mask = None
        for (lower, upper) in self.hsv_ranges:
            mask = cv2.inRange(hsv_frame, lower, upper)
            if color_mask is None:
                color_mask = mask
            else:
                color_mask += mask # Combine masks for colors like red

        # --- Contour Detection & Analysis ---
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        data = {
            "target_found": False,
            "status": "Status: No Target",
            "coords_cm": None,
            "center_px": None,
            "radius_px": None,
            "safe_circle_color": (0, 0, 255) # Default red
        }

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            
            if area > 500 and perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
                if 0.6 < circularity < 1.2:
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    center_px = (int(x), int(y))
                    radius_px = int(radius)
                    
                    data.update({
                        "target_found": True,
                        "center_px": center_px,
                        "radius_px": radius_px
                    })

                    # Localization and Status Calculation
                    self._calculate_distances(data)

        return data, output_frame, color_mask

    def _calculate_distances(self, data):
        """Helper function to calculate real-world coords and safe circle status."""
        pixel_diameter = data["radius_px"] * 2
        if pixel_diameter > 0:
            # Real-world coordinates
            dist_z = (self.known_dimension_cm * self.focal_length_px) / pixel_diameter
            dist_x = ((data["center_px"][0] - self.center_img_x) * dist_z) / self.focal_length_px
            dist_y = ((data["center_px"][1] - self.center_img_y) * dist_z) / self.focal_length_px
            data["coords_cm"] = (dist_x, dist_y, dist_z)

            # Safe circle status
            dist_from_center_px = np.sqrt((data["center_px"][0] - self.center_img_x)**2 + 
                                          (data["center_px"][1] - self.center_img_y)**2)
            
            if dist_from_center_px > self.safe_radius_px + self.tolerance_px:
                data["status"] = "Status: TOO FAR"
                data["safe_circle_color"] = (0, 255, 255) # Yellow
            else:
                data["status"] = "Status: CORRECT DISTANCE"
                data["safe_circle_color"] = (0, 255, 0) # Green

    def draw_visuals(self, frame, data):
        """Draws all the visual feedback onto the frame."""
        # Draw safe circle
        cv2.circle(frame, self.center_img, self.safe_radius_px, data["safe_circle_color"], 2)

        # Display status and coordinate text
        cv2.putText(frame, data["status"], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, data["safe_circle_color"], 2)
        coord_text = "Coords: N/A"
        if data["coords_cm"]:
            x, y, z = data["coords_cm"]
            coord_text = f"Coords: X:{x:.1f} Y:{y:.1f} Z:{z:.1f} (cm)"
        cv2.putText(frame, coord_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # If target is found, draw it
        if data["target_found"]:
            cv2.circle(frame, data["center_px"], data["radius_px"], (0, 255, 0), 4)
            cv2.circle(frame, data["center_px"], 2, (0, 0, 255), -1)

    def release(self):
        """Releases the camera and destroys all windows."""
        self.cap.release()
        cv2.destroyAllWindows()


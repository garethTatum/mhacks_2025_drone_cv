from ultralytics import YOLO
import cv2
import math

class FrontCamera:
    """
    A class to use a YOLO model for object detection, estimating the distance
    and angle of the detected object using a single camera.
    """

    def __init__(self, camera_index, model_path):
        """
        Initializes the YoloTracker.

        Args:
            camera_index (int): The index of the camera to use (e.g., 1 for the second camera).
            model_path (str): The path to the trained YOLO model weights (.pt file).
        """
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {camera_index}")

        # Get frame dimensions
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read initial frame from camera")
        self.H, self.W = frame.shape[:2]

    def _estimate_localization(self, bbox, obj_real_height, fov):
        """
        Private helper method to estimate distance and angle of an object.
        """
        x_min, y_min, x_max, y_max = bbox
        box_height = y_max - y_min
        x_center = (x_min + x_max) / 2
        
        # Focal length estimation in pixels (from FOV and image width)
        focal_length = self.W / (2 * math.tan(math.radians(fov / 2)))
        
        # Distance estimation using height of bbox
        distance = (obj_real_height * focal_length) / box_height
        
        # Angle estimation (relative to center of camera)
        img_center_x = self.W / 2
        angle = math.degrees(math.atan((x_center - img_center_x) / focal_length))
        
        return distance, angle

    def process_frame(self, frame, obj_real_height, camera_fov, conf_threshold=0.5):
        """
        Processes a single frame to find the target with the highest confidence.

        Args:
            frame (np.array): The camera frame to process.
            obj_real_height (float): The known real-world height of the object in meters.
            camera_fov (float): The horizontal field of view of the camera in degrees.
            conf_threshold (float): The minimum confidence to consider a detection.

        Returns:
            dict: A dictionary containing the detection data.
        """
        results = self.model(frame, verbose=False) # Run inference
        
        data = {
            "target_found": False,
            "distance_m": 0,
            "angle_deg": 0,
            "bbox": None,
            "confidence": 0
        }
        
        max_confidence = conf_threshold
        best_box = None

        for r in results:
            for box in r.boxes:
                confidence = box.conf[0]
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_box = box.xyxy[0]

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box)
            bbox = (x1, y1, x2, y2)
            distance, angle = self._estimate_localization(bbox, obj_real_height, camera_fov)
            
            data.update({
                "target_found": True,
                "distance_m": distance,
                "angle_deg": angle,
                "bbox": bbox,
                "confidence": max_confidence
            })

        return data

    def draw_visuals(self, frame, data):
        """Draws the bounding box and localization info onto the frame."""
        if data["target_found"]:
            x1, y1, x2, y2 = data["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            info_text = f"Dist: {data['distance_m']:.2f}m | Angle: {data['angle_deg']:.1f}deg"
            cv2.putText(frame, info_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    def release(self):
        """Releases the camera and destroys all windows."""
        self.cap.release()
        cv2.destroyAllWindows()

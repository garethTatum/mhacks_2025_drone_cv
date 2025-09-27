from ultralytics import YOLO
import cv2
import math

# Import model
model = YOLO('best.pt')

# Constants (most are placeholder)
camera_fov = 60
rsc_height = 0.114

# Localization estimation for Flight Systems
def estimate_localization(bbox, obj_real_height, fov, img_width):
    """
    Estimate distance (m) and angle (degrees) of an object from a YOLO bounding box.
    
    bbox = (x_min, y_min, x_max, y_max)
    obj_real_height = known real-world height (meters)
    fov = horizontal field of view of camera (degrees)
    img_width, img_height = image resolution
    """
    x_min, y_min, x_max, y_max = bbox
    box_height = y_max - y_min
    box_width = x_max - x_min
    x_center = (x_min + x_max) / 2
    
    # Focal length estimation in pixels (from FOV and image width)
    f = img_width / (2 * math.tan(math.radians(fov/2)))
    
    # Distance estimation using height of bbox
    distance = (obj_real_height * f) / box_height
    
    # Angle estimation (relative to center of camera)
    cx = img_width / 2
    angle = math.degrees(math.atan((x_center - cx) / f))
    
    return distance, angle

cap = cv2.VideoCapture(0)  # 0 for the default camera
while True:
    ret, frame = cap.read()  # Capture frame-by-frame

    if not ret:
        break

    results = model(frame)  # Run inference on the frame
    # Process results (e.g., draw bounding boxes, labels)
    max_confidence = 0.5 # Confidence threshold
    distance = 0
    angle = 0
    
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            
            # Localize for the max confidence
            if confidence > max_confidence:
                max_confidence = confidence

                # Localization
                distance, angle = estimate_localization(box, rsc_height, camera_fov, frame.shape[1])

            # object details
            org = [x1, y1]


    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

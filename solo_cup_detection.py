from ultralytics import YOLO

model = YOLO('yolov8n.pt')

trained_model = model.train(
    data="./red-solo-cups-4/data.yaml",
    imgsz=416,
    epochs=20,
    batch=16,
    name='rsc_detection'
)

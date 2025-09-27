from roboflow import Roboflow
rf = Roboflow(api_key="gR0P8tSHoLTvdwAeWGZ0")
project = rf.workspace("pong-yf4tt").project("red-solo-cups")
version = project.version(4)
dataset = version.download("yolov8")
                
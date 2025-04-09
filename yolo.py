#Drone tracker using OpenCV and Arduino
import cv2
import depthai as dai
import serial
import time
from ultralytics import YOLO

# Lataa YOLOv8-malli
yolo_model = YOLO("yolov8n.pt")  # 'n' (nano) versio kevyempi ja nopeampi

# DepthAI-pipeline
pipeline = dai.Pipeline()

# Kamera-asetukset
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

# XLinkOut-asetukset (kuvan siirto PC:lle)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)

# Sarjaportti yhteys Arduinoon
try:
    ArduinoSerial = serial.Serial('com3', 9600, timeout=0.1)
except serial.SerialException:
    print("Error: Could not connect to Arduino on COM3.")
    exit()

time.sleep(1)

# Pipeline käynnistys
device = dai.Device(pipeline)
video_queue = device.getOutputQueue(name="video", maxSize=8, blocking=False)

while True:
    in_frame = video_queue.get()
    frame = in_frame.getCvFrame()  # Muunna OpenCV-kehyksiin

    # YOLOv8-tunnistus
    results = yolo_model.predict(frame)

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            x_mid, y_mid = (x1 + x2) // 2, (y1 + y2) // 2

            string = 'X{0:d}Y{1:d}'.format(x_mid, y_mid)
            print(string)
            ArduinoSerial.write(string.encode('utf-8'))

            cv2.circle(frame, (x_mid, y_mid), 2, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.rectangle(frame, (640 // 2 - 30, 480 // 2 - 30),
                  (640 // 2 + 30, 480 // 2 + 30),
                  (255, 255, 255), 3)

    cv2.imshow('YOLOv8 DepthAI Video', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

del device
cv2.destroyAllWindows()

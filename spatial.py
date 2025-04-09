import cv2
import depthai as dai
import serial
import time
from ultralytics import YOLO
import numpy as np

# Lataa YOLOv8-malli
yolo_model = YOLO("Sako-Drone-Detection/yolov8n-drone.pt")

# DepthAI-pipeline
pipeline = dai.Pipeline()

# Kamera-asetukset
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # Käytetään CAM_A

# Stereo-kameran syötteet
mono_left = pipeline.create(dai.node.MonoCamera)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # Käytetään CAM_B
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

mono_right = pipeline.create(dai.node.MonoCamera)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # Käytetään CAM_C
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# StereoDepth-asetukset
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# XLinkOut-asetukset (kuvan siirto PC:lle)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)

xout_spatial = pipeline.create(dai.node.XLinkOut)
xout_spatial.setStreamName("spatial")
stereo.depth.link(xout_spatial.input)

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
spatial_queue = device.getOutputQueue(name="spatial", maxSize=8, blocking=False)

previous_positions = {}

while True:
    in_frame = video_queue.get()
    frame = in_frame.getCvFrame()  # Muunna OpenCV-kehyksiin

    # YOLOv8-tunnistus
    results = yolo_model.predict(frame)

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            x_mid, y_mid = (x1 + x2) // 2, (y1 + y2) // 2

            # Skaalaa x_mid ja y_mid vastaamaan syvyysdatan kokoa (320x240)
            scale_x = 320 / 640  # Syvyyspisteen laajennus
            scale_y = 240 / 480  # Syvyyspisteen korkeus
            x_mid_scaled = int(x_mid * scale_x)
            y_mid_scaled = int(y_mid * scale_y)

            # Oikein rajattu x_mid_scaled ja y_mid_scaled
            x_mid_scaled = np.clip(x_mid_scaled, 0, 319)  # 320 leveys
            y_mid_scaled = np.clip(y_mid_scaled, 0, 199)  # 200 korkeus

            # Spatial data (syvyyden arviointi)
            spatial_data = spatial_queue.get().getFrame()

            # Varmistetaan, että y_mid_scaled ja x_mid_scaled eivät ylitä rajaa
            if y_mid_scaled < spatial_data.shape[0] and x_mid_scaled < spatial_data.shape[1]:
                depth_value = spatial_data[y_mid_scaled, x_mid_scaled]
            else:
                depth_value = 0  # Jos koordinaatit menevät yli, käytetään syvyyttä 0

            # Liikkeen ennakointi
            obj_id = (x1, y1, x2, y2)  # Unikaalinen ID kohteelle
            if obj_id in previous_positions:
                prev_x, prev_y = previous_positions[obj_id]
                dx, dy = x_mid - prev_x, y_mid - prev_y
            else:
                dx, dy = 0, 0
            previous_positions[obj_id] = (x_mid, y_mid)

            string = 'X{0:d}Y{1:d}D{2:d}'.format(x_mid, y_mid, depth_value)
            print(string)
            ArduinoSerial.write(string.encode('utf-8'))

            cv2.circle(frame, (x_mid, y_mid), 2, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Visualisoi liikkeen suunta
            cv2.arrowedLine(frame, (x_mid, y_mid), (x_mid + dx * 3, y_mid + dy * 3), (255, 255, 0), 2)

    cv2.rectangle(frame, (640 // 2 - 30, 480 // 2 - 30),
                  (640 // 2 + 30, 480 // 2 + 30),
                  (255, 255, 255), 3)

    cv2.imshow('YOLOv8 DepthAI Video', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

del device
cv2.destroyAllWindows()


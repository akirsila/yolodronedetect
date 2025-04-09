#Drone tracker using OpenCV and Arduino
import cv2
import depthai as dai
import serial
import time

# Tunnistusmallin lataus
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)  # Tunnistus

    for x, y, w, h in faces:
        string = 'X{0:d}Y{1:d}'.format((x + w // 2), (y + h // 2))
        print(string)
        ArduinoSerial.write(string.encode('utf-8'))

        cv2.circle(frame, (x + w // 2, y + h // 2), 2, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.rectangle(frame, (640 // 2 - 30, 480 // 2 - 30),
                  (640 // 2 + 30, 480 // 2 + 30),
                  (255, 255, 255), 3)

    cv2.imshow('DepthAI Video', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

del device
cv2.destroyAllWindows()

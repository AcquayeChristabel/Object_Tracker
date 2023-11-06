import cv2
import numpy as np
import serial
import time

# Used 60 initially
CAMERA_FOV_HORIZONTAL = 90  
SERIAL_PORT = 'COM9'
BAUD_RATE = 115200
TIMEOUT = 1
CONFIDENCE_THRESHOLD = 0.5


arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)


net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")  
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


cap = cv2.VideoCapture(0)

def calculate_angle(deviation, frame_width):
    angle_per_pixel = CAMERA_FOV_HORIZONTAL / frame_width
    return deviation * angle_per_pixel

reference_center_x = None  

def send_angle_to_arduino(angle, direction):
    command = f"{direction}:{angle}\n"  
    arduino.write(command.encode('utf-8'))
    arduino.flush()  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if reference_center_x is None:
                    reference_center_x = center_x 

                deviation = center_x - reference_center_x
                angle_difference = calculate_angle(deviation, width)
                direction = 'L' if deviation < 0 else 'R'
                send_angle_to_arduino(abs(angle_difference), direction)
                print(f"Moved {abs(angle_difference):.2f} degrees to the {direction}")

              
                cv2.putText(frame, f"Moved: {abs(angle_difference):.2f} degrees {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

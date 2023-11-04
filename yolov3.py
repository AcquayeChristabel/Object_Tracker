import cv2
import numpy as np
import serial

# Establish serial connection with Arduino
arduino = serial.Serial('COM7', 115200, timeout=1)

# Function to send command to Arduino
def send_command_to_arduino(command):
    arduino.write(command.encode('utf-8'))
# Load YOLO
net = cv2.dnn.readNet(r"C:\\Users\\acqua\\Desktop\\HCI Project\\yolov3.weights", r"C:\\Users\\acqua\\Desktop\\HCI Project\\yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load the COCO class labels our YOLO model was trained on
classes = []
with open(r"C:\\Users\\acqua\\Desktop\\HCI Project\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for YOLO detection
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Calculate the deviation from the center
                deviation = center_x - (width // 2)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{classes[class_id]}: {int(confidence * 100)}%", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                # Send command to Arduino based on the deviation
                if deviation < 0:
                    # Object is to the left, move stepper to the left
                    send_command_to_arduino('b')
                elif deviation > 0:
                    # Object is to the right, move stepper to the right
                    send_command_to_arduino('a')

    # Display the resulting frame with detections
    cv2.imshow('frame', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

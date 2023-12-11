import cv2
import numpy as np
import serial
import time
from deepface import DeepFace
import pygame
from PIL import Image
import pytesseract
from gtts import gTTS
import sys
import os
# Used 60 initially
CAMERA_FOV_HORIZONTAL = 90  
SERIAL_PORT = 'COM9'
BAUD_RATE = 115200
TIMEOUT = 1
CONFIDENCE_THRESHOLD = 0.5
vertical = 'u'

arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
pygame.mixer.init()

def load_audio_files():
    pygame.mixer.init()
    return {
        "h": pygame.mixer.Sound("clap.wav"),
        "f": pygame.mixer.Sound("sad.wav"),
        "s": pygame.mixer.Sound("sad.wav"),
        "a": pygame.mixer.Sound("sad.wav"),
        "n": None
    }


audio_files = load_audio_files()

net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")  
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


cap = cv2.VideoCapture(1)

def calculate_angle(deviation, frame_width):
    angle_per_pixel = CAMERA_FOV_HORIZONTAL / frame_width
    return deviation * angle_per_pixel

reference_center_x = None  
reference_center_y = None

def analyze_mood(face_region):
    try:
        analysis = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
        return analysis['dominant_emotion'] if isinstance(analysis, dict) else analysis[0]['dominant_emotion']
    except Exception as e:
        print("Error in mood detection:", e)
        return None
def analyze_mood_from_frame(frame, net, output_layers, classes):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    dominant_emotion = None
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

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                face_region = frame[y:y+h, x:x+w]
                dominant_emotion = analyze_mood(face_region)

    return dominant_emotion

def mood_detection(frame, net, output_layers, classes):
    dominant_emotion = analyze_mood_from_frame(frame, net, output_layers, classes)
    
    if dominant_emotion:
        print(f"The dominant emotion is: {dominant_emotion[0]}")
        return dominant_emotion[0]
    else:
        print("No dominant emotion detected.")


def calculate_deviation_and_direction(center_coordinate, reference_center_coordinate, threshold=10):
    deviation = center_coordinate - reference_center_coordinate
    if abs(deviation) <= threshold:
        return deviation, 'n'  # Neutral
    elif deviation > threshold:
        return deviation, 'd'  # Down
    else:
        return deviation, 'u'  # Up
def send_angle_to_arduino(angle, direction, mood, vertical):
    command = f"{direction}:{angle}:{mood}:{vertical}\n"  
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
                if reference_center_y is None:
                    reference_center_y = center_y 

                deviation = center_x - reference_center_x
                angle_difference = calculate_angle(deviation, width)
                direction = 'R' if deviation < 0 else 'L'
                # mood = 'h'
                mood = mood_detection(frame, net, output_layers, classes)
                vertical_deviation, vertical_direction = calculate_deviation_and_direction(center_y, reference_center_y)
                # vertical_direction = 'd' 
                send_angle_to_arduino(abs(angle_difference), direction, mood, vertical_direction)
                print("+++++++++++++++++++++++++++++++++++++++++++++\n")
                print("Vertical: ", abs(angle_difference), direction, vertical_deviation, vertical_direction, mood)
                # print(f"Moved {abs(angle_difference):.2f} degrees to the {direction}")
                

              
                cv2.putText(frame, f"Moved: {abs(angle_difference):.2f} degrees {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if mood in audio_files:
                            if audio_files[mood] is not None:
                                audio_files[mood].play()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
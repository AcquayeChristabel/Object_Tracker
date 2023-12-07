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
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

pygame.mixer.init()
def capture_image_from_webcam():
    cap = cv2.VideoCapture(1)
    countdown_timer = 3  # Countdown from 5 seconds
    while countdown_timer:
        print(f"Capturing in {countdown_timer} seconds...", end="\r")
        time.sleep(1)
        countdown_timer -= 1

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        cap.release()
        return None

    img_name = "captured_document.png"
    cv2.imwrite(img_name, frame)
    print(f"\nImage captured as {img_name}")

    cap.release()
    return img_name

def setup_arduino():
    SERIAL_PORT = 'COM9'
    BAUD_RATE = 115200
    TIMEOUT = 1
    return serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)


def text_to_speech(text, language='en'):
    if not text.strip():
        
        print("No text extracted to speak.")
        return
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("output.mp3")
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()

   
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def image_to_text(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text
def load_audio_files():
    pygame.mixer.init()
    return {
        "happy": pygame.mixer.Sound("clap.wav"),
        "fear": pygame.mixer.Sound("sad.wav"),
        "sad": pygame.mixer.Sound("sad.wav"),
        "angry": pygame.mixer.Sound("sad.wav"),
        "neutral": None
    }


audio_files = load_audio_files()

def load_yolov4():
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")  
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, output_layers

def load_classes():
    with open("coco.names", "r") as f:
        return [line.strip() for line in f.readlines()]

def calculate_angle(deviation, frame_width):
    CAMERA_FOV_HORIZONTAL = 90
    return CAMERA_FOV_HORIZONTAL / frame_width * deviation

def send_data_to_arduino(arduino, angle, direction, mood):
    command = f"{direction}:{angle}:{mood}\n"  
    arduino.write(command.encode('utf-8'))
    arduino.flush()

def analyze_mood(face_region):
    try:
        analysis = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
        return analysis['dominant_emotion'] if isinstance(analysis, dict) else analysis[0]['dominant_emotion']
    except Exception as e:
        print("Error in mood detection:", e)
        return None

def mood_detection():
    arduino = setup_arduino()
    net, output_layers = load_yolov4()
    classes = load_classes()
    cap = cv2.VideoCapture(1)

    CONFIDENCE_THRESHOLD = 0.5
    reference_center_x = None

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

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    face_region = frame[y:y+h, x:x+w]
                    dominant_emotion = analyze_mood(face_region)

                    if dominant_emotion:
                        cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        if reference_center_x is None:
                            reference_center_x = center_x

                        horizontal_deviation = center_x - reference_center_x
                        horizontal_angle = calculate_angle(horizontal_deviation, width)
                        horizontal_direction = 'R' if horizontal_deviation < 0 else 'L'

                    
                        reference_center_y = height // 2  # Middle point in the vertical direction
                        vertical_deviation = center_y - reference_center_y
                        print("Vertical Stuff")
                        print(vertical_deviation, reference_center_y)
                        # vertical_direction = 'N' if vertical_deviation < 100 else 'S'
                        if vertical_deviation <= 30:
                            vertical_direction = 'N'  
                        elif vertical_deviation <= 100:
                            vertical_direction = 'M'  
                        else:
                            vertical_direction = 'S'
                        send_data_to_arduino(arduino, abs(horizontal_angle), horizontal_direction, vertical_direction, dominant_emotion[0])
                        print(f"Horizontal: {abs(horizontal_angle)}° {horizontal_direction}, Vertical: {vertical_direction}, Mood: {dominant_emotion[0]}")
                        if dominant_emotion in audio_files:
                            if audio_files[dominant_emotion] is not None:
                                audio_files[dominant_emotion].play()

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def document_processing(image_path):
    
    if not os.path.exists(image_path):
        print(f"Image file not found at {image_path}")
        return

    extracted_text = image_to_text(image_path)
    if not extracted_text.strip():  
        extracted_text = 'I am unable to read anything from this document'

    print("Extracted Text:", extracted_text)
  
    text_to_speech(extracted_text)

def main():
    if len(sys.argv) > 1:
        input_type = sys.argv[1]  
        if input_type == 'document':
        
             
            image_path = capture_image_from_webcam()
            document_processing(image_path)
           
        elif input_type == 'person':
            mood_detection()
        else:
            print("Invalid input type. Please use 'document' or 'person'.")
    else:
        print("No input type provided. Please use 'document' or 'person'.")

if __name__ == "__main__":
    main()

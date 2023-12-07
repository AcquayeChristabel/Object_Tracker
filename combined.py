import cv2  # Import the OpenCV library

import serial
import time
from PIL import Image
import pytesseract
import serial.tools.list_ports
from gtts import gTTS
import os
import time
import numpy as np
# Used 60 initially
CAMERA_FOV_HORIZONTAL = 90  
# SERIAL_PORT = 'COM9'
# BAUD_RATE = 115200
# TIMEOUT = 1
CONFIDENCE_THRESHOLD = 0.5
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def capture_image_from_webcam():
    # Initialize the webcam (use 0 as the device index to select the default webcam)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    time.sleep(1)
    ret, frame = cap.read()  # Capture a single frame
    cap.release()  # Release the webcam

    if ret:
        # If a frame is captured successfully, return it
        return frame
    else:
        # If no frame is captured, return None
        return None

def image_to_text(image):
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(image)
    return text


def perform_ocr(frame):
    if frame is not None:
        cv2.imwrite('captured_image.jpg', frame)  # Save the captured image
        extracted_text = image_to_text(frame)
        print("Extracted Text:", extracted_text)
        # text_to_speech(extracted_text)
        # send_data_to_arduino(extracted_text)
    else:
        print("Failed to capture image from webcam")

def text_to_speech(text, language='en'):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("output.mp3")
    os.system("start output.mp3")
def detect_document(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    document_contours = []

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)

        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(c)
            rect_area = w * h
            fill_ratio = area / rect_area

            # Check aspect ratio, contour area, and fill ratio
            if 0.5 < aspect_ratio < 2 and area > 1000 and 0.8 < fill_ratio < 1.0:
                document_contours.append(approx)

    return document_contours


def detect_person(frame):
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")  
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]


        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        person_detected = False
        bounding_box = None

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONFIDENCE_THRESHOLD and classes[class_id] == "person":
                    person_detected = True
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    bounding_box = (x, y, w, h)

                    # Optionally draw bounding box here or in main script
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    break  # Assuming you want to stop after detecting the first person
            if person_detected:
                break

    return person_detected, bounding_box
import cv2
import numpy as np

def decode_predictions(scores, geometry, min_confidence=0.5):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    nms_rects = []  # Initialize nms_rects here

    # Loop over the number of rows
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # Loop over the number of columns
        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

        nms_rects = []
        for (startX, startY, endX, endY) in rects:
            centerX = int((startX + endX) / 2)
            centerY = int((startY + endY) / 2)
            width = int(endX - startX)
            height = int(endY - startY)
            angle = 0  # Angle is 0 for axis-aligned boxes

            nms_rects.append(((centerX, centerY), (width, height), angle))

    return (nms_rects, confidences)

def detect_text(frame):
    # Pre-trained EAST text detector location
    east_model = "frozen_east_text_detection.pb"

    # Prepare the frame for text detection
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    # Load the pre-trained EAST text detector
    net = cv2.dnn.readNet(east_model)

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (nms_rects, confidences) = decode_predictions(scores, geometry, min_confidence=0.5)

    # Apply non-maxima suppression
    indices = cv2.dnn.NMSBoxesRotated(nms_rects, confidences, 0.5, 0.4)

    text_boxes = []
    if indices is not None and len(indices) > 0:
        # If indices is a scalar, make it a single-element list
        if isinstance(indices, (int, np.integer)):
            indices = [indices]

        # Process the indices
        for i in indices:
            # Ensure i is iterable (in case it's a single-element tuple)
            i = i if isinstance(i, tuple) else (i,)
            
            # Extract bounding box from nms_rects
            rect = nms_rects[i[0]][0]
            size = nms_rects[i[0]][1]

            # Convert to startX, startY, endX, endY format
            startX = int(rect[0] - size[0] / 2)
            startY = int(rect[1] - size[1] / 2)
            endX = int(rect[0] + size[0] / 2)
            endY = int(rect[1] + size[1] / 2)

            text_boxes.append((startX, startY, endX, endY))

    return text_boxes
from deepface import DeepFace

def detect_mood(face_frame):
    try:
        analysis = DeepFace.analyze(face_frame, actions=['emotion'])
        mood = analysis["dominant_emotion"]
        return mood
    except Exception as e:
        print(f"Error in mood detection: {e}")
        return None
def main():
    cap = cv2.VideoCapture(0)
    last_check_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        document_contours = []  # Initialize to an empty list
        person_detected = False
        person_box = None

        # Check every 10 seconds
        if current_time - last_check_time > 10:
            # Reset last check time
            last_check_time = current_time

            # Perform object detection
            person_detected, person_box = detect_person(frame)
            document_contours = detect_document(frame)

            if document_contours:
                print("Document detected")
                # Process the first detected document
                doc_contour = document_contours[0]
                x, y, w, h = cv2.boundingRect(doc_contour)
                document_frame = frame[y:y+h, x:x+w]
                extracted_text = image_to_text(document_frame)
                print("Extracted Text:", extracted_text)

            elif person_detected:
                print("Person detected")
                # Extract the face region from the frame (simplification)
                x, y, w, h = person_box
                face_frame = frame[y:y+h, x:x+w]
                mood = detect_mood(face_frame)
                if mood:
                    print(f"Mood Detected: {mood}")

        # Draw bounding boxes around detected objects for visualization
        if document_contours:
            for contour in document_contours:
                cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
        if person_detected:
            x, y, w, h = person_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Video Stream', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
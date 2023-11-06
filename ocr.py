import serial
import time
from PIL import Image
import pytesseract
import serial.tools.list_ports
from gtts import gTTS
import os
ports = list(serial.tools.list_ports.comports())
for p in ports:
    print(p)
# pytesseract.pytesseract.tesseract_cmd = (r'/usr/bin/tesseract')
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# ser = serial.Serial('COM9', 115200)  
# time.sleep(2)
def text_to_speech(text, language='en'):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("output.mp3")
    os.system("start output.mp3")
    
def image_to_text(image_path):
    # Open the image
    img = Image.open(image_path)
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(img)
    return text

def send_data_to_arduino(data):
    ser.write(data.encode())
    time.sleep(1)

if __name__ == "__main__":
    image_path = "C:\\Users\\Christabel Acquaye\\Documents\\Arduino\\sketch_sep26a\\Test.jpg"  
    extracted_text = image_to_text(image_path)
    print("Extracted Text:", extracted_text)
    text_to_speech(extracted_text)
    # send_data_to_arduino(extracted_text)

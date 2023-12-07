
# Project Setup Guide

This guide provides instructions for setting up and running the Python project.

## Prerequisites

- Python installed on your system (download from [python.org](https://www.python.org/downloads/))
- Tesseract-OCR installed (available on the [Tesseract GitHub repository](https://github.com/tesseract-ocr/tesseract))

## Installation

1. **Clone or Download the Project**: Download the project files to your local machine.

2. **Install Python Dependencies**: Navigate to the project directory in your terminal or command prompt and execute the following command:

    ```
    pip install -r requirements.txt
    ```

3. **Arduino Setup**: If your project interacts with an Arduino, make sure you have the Arduino IDE installed and the relevant Arduino script uploaded to the Arduino board.

## Running the Program

- Run the Python script using:

    ```
    python .\mood_detect.py person
    ```

    Replace `<your_script_name>` with the name of your main Python script.

## Configuration

- Adjust the serial port settings in the Python script to match the port where your Arduino is connected (`SERIAL_PORT = 'COM9'`).
- Ensure all hardware components (like webcams and Arduino) are correctly connected.
- For Windows users: If you encounter issues with `pytesseract`, verify that the path to the Tesseract executable is correctly set in the Python script.
- Note: On the first run of `deepface`, it might take additional time to download the necessary models.

## Notes

This guide should help you set up and run the project on your system. For any additional information or troubleshooting, refer to the respective documentation of the used libraries and tools.

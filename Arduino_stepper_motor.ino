#include <Arduino.h>

int dirPin = 5;
int stepPin = 6;
int stepsPerRevolution = 200; 


int dirPin2 = 3;
int stepPin2 = 4;

int minDelay = 500; 
int maxDelay = 2000; 
int accelerationSteps = 50; 

int redPin = 9;   
int greenPin = 10; 
int bluePin = 11;  

void setup() {
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);

  pinMode(stepPin2, OUTPUT);
  pinMode(dirPin2, OUTPUT);

  
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);

  Serial.begin(115200); 
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');

    int firstSeparatorIndex = input.indexOf(':');
    int secondSeparatorIndex = input.indexOf(':', firstSeparatorIndex + 1);
    int thirdSeparatorIndex = input.indexOf(':', secondSeparatorIndex + 1);

    float angle = input.substring(0, firstSeparatorIndex).toFloat();
    char horizontalDirection = input.substring(firstSeparatorIndex + 1, secondSeparatorIndex).charAt(0);
    char verticalDirection = input.substring(secondSeparatorIndex + 1, thirdSeparatorIndex).charAt(0);
    char mood = input.substring(thirdSeparatorIndex + 1).charAt(0);

    bool rotateDirection = (horizontalDirection == 'R'); 
    int steps = angleToSteps(angle);

    rotateMotor(stepPin, dirPin, steps, rotateDirection);

   
    if (verticalDirection == 'N') {
      rotateMotor(stepPin2, dirPin2, angleToSteps(30), true); 
    } else if (verticalDirection == 'S') {
      rotateMotor(stepPin2, dirPin2, angleToSteps(30), false);
    }
    // No action for 'M'

    setRGBColorBasedOnMood(mood);
  }
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    int firstSeparatorIndex = input.indexOf(':');
    int secondSeparatorIndex = input.lastIndexOf(':');
    
    char direction = input.substring(0, firstSeparatorIndex).charAt(0);
    float angle = input.substring(firstSeparatorIndex + 1, secondSeparatorIndex).toFloat();
    char verticalDirection = input.substring(secondSeparatorIndex + 1, secondSeparatorIndex + 2).charAt(0); 
    char mood = input.substring(secondSeparatorIndex + 2).charAt(0); 

    bool rotateDirection = (direction == 'R'); 
    int steps = angleToSteps(angle);

    rotateMotor(stepPin, dirPin, steps, rotateDirection);

    
    if (verticalDirection == 'N') {
      rotateMotor(stepPin2, dirPin2, angleToSteps(30), true); 
    } else if (verticalDirection == 'S') {
      rotateMotor(stepPin2, dirPin2, angleToSteps(30), false); 
    }

    setRGBColorBasedOnMood(mood);
  }
}
void rotateMotor(int stepPin, int dirPin, int steps, bool direction) {
  digitalWrite(dirPin, direction ? HIGH : LOW);
  int stepDelay = maxDelay;
  int stepChange = (maxDelay - minDelay) / accelerationSteps;

  for (int i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(stepDelay);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(stepDelay);

    if (i < accelerationSteps && stepDelay > minDelay) {
      stepDelay -= stepChange;
    } else if (i >= steps - accelerationSteps && stepDelay < maxDelay) {
      stepDelay += stepChange;
    }
  }
}
int angleToSteps(float angle) {
  return (int)(angle / 360.0 * stepsPerRevolution);
}

void setRGBColorBasedOnMood(char mood) {
  switch (mood) {
    case 'a': // Anger
      analogWrite(redPin, 255);   // Red
      analogWrite(greenPin, 0);
      analogWrite(bluePin, 0);
      break;
    case 'd': // Disgust
      analogWrite(redPin, 255);   // Red
      analogWrite(greenPin, 0);
      analogWrite(bluePin, 0);
      break;
    case 'f': // Fear
      analogWrite(redPin, 255);   // Red
      analogWrite(greenPin, 0);
      analogWrite(bluePin, 0);
      break;
    case 'h': // Happiness
      analogWrite(redPin, 255);   // Yellow
      analogWrite(greenPin, 255);
      analogWrite(bluePin, 0);
      break;
    case 's': // Sadness
      analogWrite(redPin, 255);   // Red
      analogWrite(greenPin, 0);
      analogWrite(bluePin, 0);  // Blue
      break;
    case 'u': // Surprise
      analogWrite(redPin, 255);   // Cyan
      analogWrite(greenPin, 255);
      analogWrite(bluePin, 255);
      break;
    case 'n': // Neutral
      analogWrite(redPin, 128);   // White
      analogWrite(greenPin, 128);
      analogWrite(bluePin, 128);
      break;
    default:  // Default color
      analogWrite(redPin, 0);   
      analogWrite(greenPin, 0);
      analogWrite(bluePin, 0);    // Turn off LED
  }
}

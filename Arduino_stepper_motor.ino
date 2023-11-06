#include <Arduino.h>

int dirPin = 5;
int stepPin = 6;
int stepsPerRevolution = 200; 

int minDelay = 500; 
int maxDelay = 2000; 
int accelerationSteps = 50; 

void setup() {
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  Serial.begin(115200); 
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    int separatorIndex = input.indexOf(':');
    char direction = input.substring(0, separatorIndex).charAt(0);
    float angle = input.substring(separatorIndex + 1).toFloat();

    bool rotateDirection = (direction == 'R'); 
    int steps = angleToSteps(angle);

    rotateMotor(steps, rotateDirection);
  }
}

void rotateMotor(int steps, bool direction) {
  digitalWrite(dirPin, direction ? HIGH : LOW); // Set the direction
  int stepDelay = maxDelay;
  int stepChange = (maxDelay - minDelay) / accelerationSteps;

  for (int i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(stepDelay);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(stepDelay);

   
    if (i < accelerationSteps && stepDelay > minDelay) {
      stepDelay -= stepChange;
    }
    
    else if (i >= steps - accelerationSteps && stepDelay < maxDelay) {
      stepDelay += stepChange;
    }
  }
}

int angleToSteps(float angle) {
  return (int)(angle / 360.0 * stepsPerRevolution);
}

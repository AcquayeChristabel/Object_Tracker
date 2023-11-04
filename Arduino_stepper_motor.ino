int dirPin = 6;
int stepPin = 5;
const int stepsPerRevolution = 200;  // for your motor
const int stepsFor90Degrees = stepsPerRevolution / 4;  // Calculate steps needed for 90 degrees
const int stepsFor180Degrees = stepsPerRevolution / 2;  // Calculate steps needed for 180 degrees

// Adjust the delay for slower rotation. You can adjust this value as needed.
const int delayBetweenSteps = 1000;  // 5 milliseconds

void setup() {
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  
  Serial.begin(115200);  // Initialize serial communication at 9600 baud
}

void loop() {
  if (Serial.available()) {  // Check if data is available to read from the serial port
    char command = Serial.read();  // Read a character from the serial port
    
    switch (command) {
      case 'a':  // Rotate 90 degrees to the right (clockwise)
        rotate(stepsFor90Degrees, HIGH);
        break;
      case 'b':  // Rotate 90 degrees to the left (counterclockwise)
        rotate(stepsFor90Degrees, LOW);
        break;
      case 'x':  // Rotate 180 degrees to the left (counterclockwise)
        rotate(stepsFor180Degrees, LOW);
        break;
      case 'y':  // Rotate 180 degrees to the right (clockwise)
        rotate(stepsFor180Degrees, HIGH);
        break;
    }
  }
}

void rotate(int steps, int direction) {
  digitalWrite(dirPin, direction);
  
  for (int x = 0; x < steps; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(delayBetweenSteps);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(delayBetweenSteps);
  }
  delay(1000);  // Wait a second after rotation
}
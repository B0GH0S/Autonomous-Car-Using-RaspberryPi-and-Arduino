#include <AFMotor.h>
// Define motor pins
AF_DCMotor motor1(1);  // Motor 1 connected to port M1
AF_DCMotor motor2(2);  // Motor 2 connected to port M2
AF_DCMotor motor3(3);  // Motor 3 connected to port M3
AF_DCMotor motor4(4);  // Motor 4 connected to port M4

// Define initial speed
const int INITIAL_SPEED = 200;
int currentSpeed = INITIAL_SPEED;

void setup() {
  // Initialize motors with speed 200
  motor1.setSpeed(INITIAL_SPEED);
  motor2.setSpeed(INITIAL_SPEED);
  motor3.setSpeed(INITIAL_SPEED);
  motor4.setSpeed(INITIAL_SPEED);
  Serial.begin(115200);
}

void loop() {
  if (Serial.available()) {
    // Read command from serial port
    char command = Serial.read();
    
    if (command == 'w') {
      moveForward();
    }
    else if (command == 'a') {
      turnLeft();
    }
    else if (command == 's') {
      moveBackward();
    }
    else if (command == 'd') {
      turnRight();
    }
    else if (command == 'x') {
      stop();
    }
    // Speed control (0-9)
    else if (command >= '0' && command <= '9') {
      int speedPercent = (command - '0') * 25; // Convert to range 0-225
      adjustSpeed(speedPercent);
    }
  }
}

void adjustSpeed(int newSpeed) {
  currentSpeed = newSpeed;
  motor1.setSpeed(currentSpeed);
  motor2.setSpeed(currentSpeed);
  motor3.setSpeed(currentSpeed);
  motor4.setSpeed(currentSpeed);
}

void moveForward() {
  motor1.run(BACKWARD);
  motor2.run(BACKWARD);
  motor3.run(BACKWARD);
  motor4.run(BACKWARD);
}

void moveBackward() {
  motor1.run(FORWARD);
  motor2.run(FORWARD);
  motor3.run(FORWARD);
  motor4.run(FORWARD);
}

void turnLeft() {
  motor1.run(BACKWARD);
  motor2.run(BACKWARD);
  motor3.run(FORWARD);
  motor4.run(FORWARD);
}

void turnRight() {
  motor3.run(BACKWARD);
  motor4.run(BACKWARD);
  motor1.run(FORWARD);
  motor2.run(FORWARD);
}

void stop() {
  motor1.run(RELEASE);
  motor2.run(RELEASE);
  motor3.run(RELEASE);
  motor4.run(RELEASE);
}

/*---------------Includes-----------------------------------*/
#include <math.h>

/*---------------Module Defines-----------------------------*/
#define OFF               0
#define FORWARD           HIGH
#define REVERSE           LOW
#define GROWTH            HIGH
#define RETRACT           LOW

/*---------------Pin Declares-------------------------------*/
int motorPin    = 5;  // PWM output pin for rewinding motor
int motorDirPin = 8;  // Direction output pin for motor

/*---------------Motor Dependent Parameters-----------------*/
//  *** TUNE THESE VALUES TO MATCH YOUR MOTOR ***
int minGrowthSpeed  = 150; // Minimum PWM speed to start growing  (0-255)
int minGrowthStop   = 0;   // PWM value to stop motor
int minRetractSpeed = 150; // Minimum PWM speed to start retracting (0-255)

/*---------------Control Variables--------------------------*/
int motorSpeed   = 150;    // Default motor speed (0-255)
int speedDelta   = 15;     // How much speed changes per D/A key press

bool vineState  = GROWTH;  // Is the motor growing or retracting?
 
unsigned char theKey = ' ';

/*---------------Module Function Prototypes--------------*/
unsigned char TestForKey(void);
void RespToKey(void);
void grow(int);
void retract(int);
void halt();
void deflate();
void inflate();

/*---------------Pre-formed Vine Robot Setup ----------------------*/
void setup() {
  Serial.begin(9600);

  // Output pins
  pinMode(motorPin, OUTPUT);
  pinMode(motorDirPin, OUTPUT);

  // Initialize motor — start stopped
  analogWrite(motorPin, OFF);
  digitalWrite(motorDirPin, FORWARD);

  Serial.println("VINEAR Ready. Commands: W=Grow, S=Retract, D=Faster, A=Slower, SPACE or X=Stop");
}

/*---------------Pre-formed Vine Robot Main Loop-------------------*/
void loop() {

  // Only read key if data is available (avoids reading -1 junk)
  if (Serial.available() > 0) {
    theKey = Serial.read();

    if (theKey == 'W' || theKey == 'w') {
      //GROW, MOVE FORWARD
      grow(motorSpeed);
      vineState = GROWTH;
      Serial.println("Growing...");
    }

    else if (theKey == 'S' || theKey == 's') {
      //RETRACT, MOVE BACK
      retract(motorSpeed);
      vineState = RETRACT;
      Serial.println("Retracting...");
    }

    else if (theKey == 'D' || theKey == 'd') {
      //INCREASE SPEED
      motorSpeed += speedDelta;
      motorSpeed = constrain(motorSpeed, 0, 255); // Clamp between 0-255
      Serial.print("Speed increased to: ");
      Serial.println(motorSpeed);

      // Apply new speed immediately if already moving
      if (vineState == GROWTH) grow(motorSpeed);
      else retract(motorSpeed);
    }

    else if (theKey == 'A' || theKey == 'a') {
      //DECREASE SPEED
      motorSpeed -= speedDelta;
      motorSpeed = constrain(motorSpeed, 0, 255); // Clamp between 0-255
      Serial.print("Speed decreased to: ");
      Serial.println(motorSpeed);

      // Apply new speed immediately if already moving
      if (vineState == GROWTH) grow(motorSpeed);
      else retract(motorSpeed);
    }

    else if (theKey == ' ') {
      //MANUAL STOP via keyboard
      halt();
      Serial.println("MANUAL STOP");
    }

    else if (theKey == 'X' || theKey == 'x') {
      // *** AI STOP — triggered by Python human detector ***
      halt();
      Serial.println("AI STOP: Human detected! Vine halted.");
    }
  }
}


/*----------------------Module Functions--------------------------*/
unsigned char TestForKey(void) {
  unsigned char KeyEventOccurred;
  KeyEventOccurred = Serial.available();
  return KeyEventOccurred;
}

// Tells the vine robot to grow at a particular motor speed
void grow(int speed) {
  analogWrite(motorPin, speed);
  digitalWrite(motorDirPin, FORWARD);
}

// Tells the vine robot to retract at a particular motor speed
void retract(int speed) {
  analogWrite(motorPin, speed);
  digitalWrite(motorDirPin, REVERSE);
}

// Stops the vine robot's growth and remains pressurized
void halt() {
  analogWrite(motorPin, minGrowthStop);
}

// Pressurizes the chamber
void deflate() {
}

// Deflates the vine robot
void inflate() {
}

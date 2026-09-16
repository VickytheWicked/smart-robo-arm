#include <Adafruit_PWMServoDriver.h>
#include <Arduino.h>

// ============================================================
// SERVO STRUCTURES
// ============================================================

struct Servo {
  int channel;
  int max;
  int min;
  int currentAngle;
};

struct states {
  int ind;
  int angle;
};

// ============================================================
// ORIGINAL / RESET POSITIONS
// ============================================================

states original[6] = {{0, 30}, {1, 180}, {2, 0}, {3, 40}, {4, 90}, {5, 0}};

// ============================================================
// SERVO CONFIGURATION
// ============================================================

const int servo_num = 6;

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo indexes:
//
// 0 = Base
// 1 = Shoulder
// 2 = Elbow
// 3 = Wrist
// 4 = Wrist rotation
// 5 = Gripper

Servo all_servo[6] = {{0, 540, 85, 31},  {1, 520, 90, 179}, {2, 520, 190, 1},
                      {3, 400, 100, 51}, {4, 550, 100, 91}, {5, 400, 80, 1}};

// ============================================================
// GLOBAL VARIABLES
// ============================================================

int targetAngle = -1;
int targetServo = -1;

String state = "ALIGN";

int speed = 8;

// ============================================================
// TABLE CALIBRATION
// ============================================================
//
// IMPORTANT:
//
// These are RAW ESP32 servo angles.
//
// Format:
//
//     { Elbow, Shoulder }
//
// Each pair represents the experimentally observed
// table-contact boundary.
//
// Your dashboard uses inverted values, but that inversion
// is NOT handled here.
//
// RAW examples:
//
//     Elbow = 150 -> Shoulder = 75
//     Elbow = 100 -> Shoulder = 120
//     Elbow = 20  -> Shoulder = 160
//
// ============================================================

const int TABLE_POINTS[][2] = {

    {180, 45}, {170, 55},  {160, 65},  {150, 75},  {140, 85},
    {130, 95}, {120, 105}, {110, 115}, {100, 120}, {90, 125},
    {80, 130}, {70, 135},  {60, 140},  {50, 145},  {40, 150},
    {30, 155}, {20, 160},  {10, 165},  {0, 170}};

// const int TABLE_POINTS[][2] = {

//   {45,  180},
//   {55,  170},
//   {65,  160},
//   {75,  150},
//   {85,  140},
//   {95,  130},
//   {105, 120},
//   {120, 100},
//   {125, 90},
//   {130, 80},
//   {135, 70},
//   {140, 60},
//   {145, 50},
//   {150, 40},
//   {155, 30},
//   {160, 20},
//   {165, 10},
//   {170, 0}

// };

const int TABLE_POINT_COUNT = sizeof(TABLE_POINTS) / sizeof(TABLE_POINTS[0]);

// ============================================================
// TABLE CONTACT TOLERANCE
// ============================================================
//
// Start with 0 for the first physical test.
//
// If the arm consistently needs 1-2 degrees of tolerance,
// change this to 1 or 2.
//
// ============================================================

const int TABLE_TOLERANCE = 9;

// ============================================================
// SERVO ANGLE SET
// ============================================================

void angleSet() {
  int flag = 0;

  if (targetServo >= 0) {
    Servo &s = all_servo[targetServo];

    int step = 1;

    // --------------------------------------------------------
    // Gripper
    // --------------------------------------------------------

    if (targetServo == 5)
      step = 2;

    // --------------------------------------------------------
    // Wrist rotation
    // --------------------------------------------------------

    if (targetServo == 4)
      step = 2;

    // --------------------------------------------------------
    // Wrist
    // --------------------------------------------------------

    if (targetServo == 3)
      step = 1;

    // --------------------------------------------------------
    // Elbow
    // --------------------------------------------------------

    if (targetServo == 2)
      step = 1;

    // --------------------------------------------------------
    // Shoulder
    // --------------------------------------------------------

    if (targetServo == 1)
      step = 1;

    // --------------------------------------------------------
    // Base
    // --------------------------------------------------------

    if (targetServo == 0)
      step = 1;

    // --------------------------------------------------------
    // Determine direction
    // --------------------------------------------------------

    if (s.currentAngle < targetAngle) {
      flag = 1;
    } else if (s.currentAngle > targetAngle) {
      flag = 0;
    }

    // --------------------------------------------------------
    // Increase angle
    // --------------------------------------------------------

    while (s.currentAngle < targetAngle && flag == 1) {
      s.currentAngle += step;

      s.currentAngle = constrain(s.currentAngle, 0, 180);

      int safeMin = s.min + 5;
      int safeMax = s.max - 5;

      int pulse = map(s.currentAngle, 0, 180, safeMin, safeMax);

      pwm.setPWM(s.channel, 0, pulse);

      delay(20);
    }

    // --------------------------------------------------------
    // Decrease angle
    // --------------------------------------------------------

    while (s.currentAngle > targetAngle && flag == 0) {
      s.currentAngle -= step;

      s.currentAngle = constrain(s.currentAngle, 0, 180);

      int safeMin = s.min + 5;
      int safeMax = s.max - 5;

      int pulse = map(s.currentAngle, 0, 180, safeMin, safeMax);

      pwm.setPWM(s.channel, 0, pulse);

      delay(20);
    }
  }
}

// ============================================================
// GET CALIBRATED TABLE SHOULDER
// ============================================================
//
// Input:
//
//     current RAW elbow angle
//
// Output:
//
//     expected RAW shoulder angle at the calibrated
//     table-contact boundary.
//
// Linear interpolation is used between measured points.
//
// Example:
//
//     Elbow 150 -> Shoulder 75
//     Elbow 140 -> Shoulder 85
//
//     Elbow 145 -> Shoulder 80
//
// ============================================================

bool tableReached() {
  int shoulder = all_servo[1].currentAngle;
  int elbow = all_servo[2].currentAngle;

  // IMPORTANT: table calibration is DISCRETE.
  // Never interpolate between measured points.
  for (int i = 0; i < TABLE_POINT_COUNT; i++) {
    int calibratedElbow = 180 - TABLE_POINTS[i][0];
    int calibratedShoulder = 180 - TABLE_POINTS[i][1];

    // if (elbow == calibratedElbow &&
    //     shoulder == calibratedShoulder)
    if (abs(elbow - calibratedElbow) <= TABLE_TOLERANCE &&
        abs(shoulder - calibratedShoulder) <= TABLE_TOLERANCE) {
      Serial.print("TABLE CONTACT POSITION -> Shoulder: ");
      Serial.print(shoulder);
      Serial.print(" Elbow: ");
      Serial.println(elbow);
      return true;
    }
  }

  return false;
}

// ============================================================
// FIND NEAREST CALIBRATED TABLE POINT
// ============================================================
//
// The current shoulder position represents the horizontal
// position reached by the existing LEFT/RIGHT/FORWARD/BACK
// servoing. We select an ACTUAL measured calibration point.
// There is NO interpolation.
//
// Returns the index of the calibration point whose shoulder
// angle is closest to the current shoulder angle.
// ============================================================

int findNearestTablePoint(int currentShoulder, int currentElbow) {
  // Only consider calibration points that continue the existing
  // DOWN direction:
  //     shoulder decreases
  //     elbow increases
  // This prevents CENTERED from commanding the arm back upward.
  int bestIndex = -1;
  int bestDifference = 181;

  for (int i = 0; i < TABLE_POINT_COUNT; i++) {
    int calibratedElbow = TABLE_POINTS[i][1];
    int calibratedShoulder = TABLE_POINTS[i][0];

    if (calibratedShoulder <= currentShoulder &&
        calibratedElbow >= currentElbow) {
      int difference = abs(currentShoulder - calibratedShoulder);

      if (difference < bestDifference) {
        bestDifference = difference;
        bestIndex = i;
      }
    }
  }

  return bestIndex;
}

// ============================================================
// MOVE TO CALIBRATED TABLE POINT
// ============================================================
//
// Once CENTERED is reached, the arm does NOT continue blindly
// along the old equal-and-opposite DOWN path.
//
// Instead, it moves to the nearest experimentally measured
// table-contact configuration. This uses only the calibration
// table; no FK and no interpolation are involved.
// ============================================================

void moveToTablePoint() {
  int currentShoulder = all_servo[1].currentAngle;
  int currentElbow = all_servo[2].currentAngle;

  int pointIndex = findNearestTablePoint(currentShoulder, currentElbow);

  if (pointIndex < 0) {
    Serial.println("NO FORWARD CALIBRATION POINT FOUND");
    return;
  }

  int targetElbow = TABLE_POINTS[pointIndex][1];
  int targetShoulder = TABLE_POINTS[pointIndex][0];

  Serial.print("CALIBRATED TABLE TARGET -> Shoulder: ");
  Serial.print(targetShoulder);
  Serial.print(" Elbow: ");
  Serial.println(targetElbow);

  // Move elbow to the measured calibration value.
  if (currentElbow != targetElbow) {
    targetServo = 2;
    targetAngle = targetElbow;
    angleSet();
  }

  // Move shoulder to the measured calibration value.
  if (currentShoulder != targetShoulder) {
    targetServo = 1;
    targetAngle = targetShoulder;
    angleSet();
  }

  // Verify exact calibrated position.
  if (tableReached()) {
    Serial.println("TABLE REACHED -> GRAB");
    state = "GRAB";
  } else {
    Serial.println("TABLE TARGET NOT REACHED");
  }
}

// ============================================================
// MOVE LEFT
// ============================================================

void moveLeft(int i) {
  targetServo = 0;

  targetAngle = all_servo[targetServo].currentAngle + i;

  angleSet();
}

// ============================================================
// MOVE RIGHT
// ============================================================

void moveRight(int i) {
  targetServo = 0;

  targetAngle = all_servo[targetServo].currentAngle - i;

  angleSet();
}

// ============================================================
// MOVE FORWARD
// ============================================================

void moveForward(int i) {
  targetServo = 1;

  targetAngle = all_servo[targetServo].currentAngle - i;

  angleSet();
}

// ============================================================
// MOVE BACK
// ============================================================

void moveBack(int i) {
  targetServo = 1;

  targetAngle = all_servo[targetServo].currentAngle + i;

  angleSet();
}

// ============================================================
// EXTEND ARM
// ============================================================

void extendArm(int i) {
  targetServo = 1;

  targetAngle = all_servo[targetServo].currentAngle + i;

  angleSet();
}

// ============================================================
// RETRACT ARM
// ============================================================

void retractArm(int i) {
  targetServo = 2;

  targetAngle = all_servo[targetServo].currentAngle - i;

  angleSet();
}

// ============================================================
// MOVE DOWN
// ============================================================
//
// CENTERED calls this function.
//
// Starting:
//
//     Shoulder = 180
//     Elbow    = 0
//
// Movement:
//
//     Shoulder decreases
//     Elbow increases
//
// The old implementation stopped ONLY at:
//
//     Shoulder = 160
//     Elbow    = 20
//
// This version instead uses the complete calibrated table.
//
// ============================================================

void moveDown(int i) {
  // First check whether we are already exactly on one of the
  // experimentally measured table-contact configurations.
  if (tableReached()) {
    Serial.println("TABLE REACHED -> GRAB");
    state = "GRAB";
    return;
  }

  // Preserve the original DOWN behaviour for the requested step.
  // No interpolated table boundary is checked during this move.
  int currentElbow = all_servo[2].currentAngle;
  int currentShoulder = all_servo[1].currentAngle;

  int stepAmount = i;
  if (stepAmount <= 0)
    stepAmount = 1;

  int nextElbow = constrain(currentElbow + stepAmount, 0, 180);

  int nextShoulder = constrain(currentShoulder - stepAmount, 0, 180);

  Serial.print("DOWN -> Shoulder: ");
  Serial.print(nextShoulder);
  Serial.print("  Elbow: ");
  Serial.println(nextElbow);

  targetServo = 2;
  targetAngle = nextElbow;
  angleSet();

  targetServo = 1;
  targetAngle = nextShoulder;
  angleSet();

  // Only an EXACT calibration point can trigger the grab.
  if (tableReached()) {
    Serial.println("TABLE REACHED -> GRAB");
    state = "GRAB";
  }
}

// ============================================================
// MOVE UP
// ============================================================

void moveUp(int i) {
  targetServo = 2;

  targetAngle = all_servo[targetServo].currentAngle - i;

  angleSet();

  targetServo = 1;

  targetAngle = all_servo[targetServo].currentAngle + i;

  angleSet();
}

// ============================================================
// OPEN GRIPPER
// ============================================================

void open() {
  targetServo = 5;

  targetAngle = 0;

  angleSet();
}

// ============================================================
// CLOSE GRIPPER
// ============================================================

void close() {
  targetServo = 5;

  targetAngle = 180;

  angleSet();
}

// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(115200);

  Serial.setTxTimeoutMs(0);

  // ----------------------------------------------------------
  // Initialize PCA9685.
  // ----------------------------------------------------------

  pwm.begin();

  pwm.setPWMFreq(50);

  // ----------------------------------------------------------
  // Move all servos to original positions.
  // ----------------------------------------------------------

  for (int i = 0; i < 6; i++) {
    targetServo = original[i].ind;

    targetAngle = original[i].angle;

    angleSet();
  }

  delay(200);

  Serial.println();

  Serial.println("================================");

  Serial.println("       ROBOTIC ARM READY");

  Serial.println("================================");

  Serial.println("TABLE CALIBRATION: "
                 "19 POINTS");

  Serial.println("RAW CALIBRATION:");

  Serial.println("Elbow -> Shoulder");

  for (int i = 0; i < TABLE_POINT_COUNT; i++) {
    Serial.print(TABLE_POINTS[i][0]);

    Serial.print(" -> ");

    Serial.println(TABLE_POINTS[i][1]);
  }

  Serial.println("--------------------------------");

  Serial.println("Dashboard uses inverted angles.");

  Serial.println("Slider inversion is NOT applied");

  Serial.println("inside ESP32 calibration.");

  Serial.println("================================");
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {
  String cmd;

  if (Serial.available()) {
    cmd = Serial.readStringUntil('\n');

    cmd.trim();

    // ========================================================
    // DIRECT SERVO COMMAND
    //
    // S1:160
    // S2:20
    //
    // These are RAW ESP32 angles.
    // ========================================================

    if (cmd.startsWith("S") && cmd.indexOf(':') > 0) {
      int col = cmd.indexOf(':');

      int idx = cmd.substring(1, col).toInt();

      int ang = constrain(cmd.substring(col + 1).toInt(), 0, 180);

      if (idx >= 0 && idx < servo_num) {
        targetServo = idx;

        targetAngle = ang;

        angleSet();

        Serial.print("OK S");

        Serial.print(idx);

        Serial.print(":");

        Serial.println(all_servo[idx].currentAngle);
      }
    }

    // ========================================================
    // NORMAL ALIGNMENT COMMANDS
    // ========================================================

    else if (state == "ALIGN" && cmd.indexOf(':') > 0) {
      String direction = cmd.substring(0, cmd.indexOf(':'));

      speed = cmd.substring(cmd.indexOf(':') + 1).toInt();

      Serial.print("Command: ");

      Serial.println(direction);

      // ------------------------------------------------------
      // LEFT
      // ------------------------------------------------------

      if (direction == "LEFT") {
        moveLeft(speed);
      }

      // ------------------------------------------------------
      // RIGHT
      // ------------------------------------------------------

      else if (direction == "RIGHT") {
        moveRight(speed);
      }

      // ------------------------------------------------------
      // FORWARD
      // ------------------------------------------------------

      else if (direction == "FORWARD") {
        moveForward(speed);
      }

      // ------------------------------------------------------
      // BACK
      // ------------------------------------------------------

      else if (direction == "BACK") {
        moveBack(speed);
      }

      // ------------------------------------------------------
      // CENTERED
      //
      // This is the ONLY automatic path toward the table.
      // ------------------------------------------------------

      else if (direction == "CENTERED") {
        // CENTERED means XY alignment is complete.
        // Finish the approach using a measured calibration point.
        // moveToTablePoint();
        moveDown(speed);
      }
    }

    // ========================================================
    // MANUAL CATCH
    // ========================================================
    //
    // Kept for compatibility.
    //
    // Python will no longer automatically send CATCH based
    // on bounding-box area.
    //
    // ========================================================

    else if (cmd == "CATCH") {
      state = "GRAB";
    }

    // ========================================================
    // GRAB STATE
    // ========================================================

    if (state == "GRAB") {
      Serial.println("GRASPING");

      close();

      delay(500);

      delay(300);

      state = "DONE";

      speed = 8;

      Serial.println("GRASP COMPLETE");
    }

    // ========================================================
    // RESET
    // ========================================================

    if (cmd == "RESET") {
      state = "ALIGN";

      for (int i = 0; i < 6; i++) {
        targetServo = original[i].ind;

        targetAngle = original[i].angle;

        angleSet();
      }

      Serial.println("ARM RESET");
    }

    delay(20);
  }
}
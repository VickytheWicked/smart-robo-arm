#include <Arduino.h>

#include <Adafruit_PWMServoDriver.h>

struct Servo
{
  int channel;
  int max;
  int min;
  int currentAngle;
};

struct states
{
  int ind;
  int angle;

};

states original[6]={{0,1},{1,100},{2,90},{3,90},{4,90},{5,0}};



// -------- SERVOS --------
// Servo baseServo;      // rotation
// Servo shoulderServo;  // forward/back
// Servo elbowServo;     // extension
// Servo wristServo;     // pitch
// Servo wristRollServo; // roll (optional)
// Servo gripperServo;   // grab

const int servo_num = 6;
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

Servo all_servo[6] = {{0, 540, 85, 0},{1, 520, 90, 111},{2, 520, 190, 91}, {3, 400, 100, 91}, {4, 550, 100, 91}, {5, 400, 80,1}};
int targetAngle = -1;
int targetServo = -1;

// // -------- ULTRASONIC --------
// #define TRIG D7
// #define ECHO D8



// -------- STATE --------
String state = "ALIGN";


void angleSet()
{
  int flag=0;
  if (targetServo >= 0)
  {
    Servo &s = all_servo[targetServo];

    int step = 1;
    if (targetServo == 5)
      step = 2; // gripper
    if (targetServo == 4)
      step = 2; // wrist roll
    if (targetServo == 3)
      step = 1; // wrist pitch
    if (targetServo == 2)
      step = 1; // elbow
    if (targetServo == 1)
      step = 1; // shoulder
    if (targetServo == 0)
      step = 1; // base slow

    if (s.currentAngle < targetAngle)
    {
      flag=1;
    }
    else if (s.currentAngle > targetAngle)
    {
      flag=0;
    }
      

    while (s.currentAngle < targetAngle && flag==1)
    {
      s.currentAngle += step;
      s.currentAngle = constrain(s.currentAngle, 0, 180);

      int safeMin = s.min + 5;
      int safeMax = s.max - 5;

      int pulse = map(s.currentAngle, 0, 180, safeMin, safeMax);
      pwm.setPWM(s.channel, 0, pulse);
      delay(20);

    }
    
    while (s.currentAngle > targetAngle && flag==0)
    {
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
// -------- SETUP --------
void setup() {
  Serial.begin(115200);

  pwm.begin();
  pwm.setPWMFreq(50); // 50 Hz for servo 

  // pinMode(TRIG, OUTPUT);
  // pinMode(ECHO, INPUT);

  for(int i=0;i<6;i++)
  {
    targetServo=original[i].ind;
    targetAngle=original[i].angle;
    angleSet();
  }

  delay(200);
}

// -------- ULTRASONIC --------
// float getDistance() {
//   digitalWrite(TRIG, LOW);
//   delayMicroseconds(2);

//   digitalWrite(TRIG, HIGH);
//   delayMicroseconds(10);
//   digitalWrite(TRIG, LOW);

//   long duration = pulseIn(ECHO, HIGH);
//   return duration * 0.034 / 2;
// }

// -------- MOVEMENT --------
void moveLeft()  
{ 
  targetServo=0;
  targetAngle=all_servo[targetServo].currentAngle+2;

  
  angleSet(); 
}
void moveRight()
{
  targetServo=0;
  targetAngle=all_servo[targetServo].currentAngle-2;

  
  angleSet();  
}

void moveForward() 
{ 
  targetServo=1;
  targetAngle=all_servo[targetServo].currentAngle-2;
  angleSet(); 
}
void moveBack()  
{
  targetServo=1;
  targetAngle=all_servo[targetServo].currentAngle+2;

  

  angleSet();  
}

void extendArm()
{
  targetServo=1;
  targetAngle=all_servo[targetServo].currentAngle+1;
  angleSet(); 
}
void retractArm()  
{ 
  targetServo=2;
  targetAngle=all_servo[targetServo].currentAngle-1;
  angleSet();  
}

void moveDown() {
//adjust elbow=1 and shoulder=2
  targetServo=2;
  targetAngle=all_servo[targetServo].currentAngle+2;
  angleSet();
  targetServo=1;
  targetAngle=all_servo[targetServo].currentAngle-2;
  angleSet();
}

void moveUp() {
  targetServo=2;
  targetAngle=all_servo[targetServo].currentAngle-2;
  angleSet();
  targetServo=1;
  targetAngle=all_servo[targetServo].currentAngle+2;
  angleSet();
}

void open()
{
  targetServo=5;
  targetAngle=0;
  angleSet();
}

void close()
{
  targetServo=5;
  targetAngle=180;
  angleSet();
}


// -------- LOOP --------
void loop()
{
  String cmd;
  // Serial.println("Waiting for command...");
  
  if (Serial.available())
  {
    cmd = Serial.readStringUntil('\n');
    cmd.trim();

    Serial.print("Command: ");
    Serial.println(cmd);
    
    

    // Direct servo command: S<index>:<angle>  e.g. S0:90
    if (cmd.startsWith("S") && cmd.indexOf(':') > 0)
    {
      int col = cmd.indexOf(':');
      int idx = cmd.substring(1, col).toInt();
      int ang = constrain(cmd.substring(col + 1).toInt(), 0, 180);
      if (idx >= 0 && idx < servo_num)
      {
        targetServo = idx;
        targetAngle = ang;
        angleSet();
        Serial.print("OK S");
        Serial.print(idx);
        Serial.print(":");
        Serial.println(all_servo[idx].currentAngle);
      }
    }
    else if (state == "ALIGN")
    {
      
      // Serial.print("State: ");
      // Serial.println(state);

      if (cmd == "LEFT")
        moveLeft();
      else if (cmd == "RIGHT")
        moveRight();
      else if (cmd == "FORWARD")
        moveForward();
      else if (cmd == "BACK")
        moveBack();

      else if (cmd == "CENTERED")
      {
        // state = "APPROACH";
        moveDown();
      }
      else if (cmd=="CATCH")
      {
        state = "GRAB";

      }
    }

    //-------- APPROACH --------
    // if (state == "APPROACH" && cmd == "CENTERED")
    // {
    //   // float d = getDistance();
    //   // Serial.print("Distance: ");
    //   // Serial.println(d);

    //   moveDown();

    //   // else
    //   // {
    //   //   state = "GRAB";
    //   // }
    // }

    // -------- GRAB --------
    if (state == "GRAB")
    {
      close();
      delay(500);

      moveUp(); // lift slightly
      delay(300);

      state = "DONE";
    }

    if(cmd=="RESET")
    {
      state="ALIGN";
    }
    // -------- DONE --------
    // if (state == "DONE")
    // {
    //   // hold position
    // }
    // if (cmd != "CENTERED")
    // {
    //   state = "ALIGN";
    // }

    // Serial.print("TargetServo: ");
    // Serial.println(targetServo);
    // Serial.print("Target Angle: ");
    // Serial.println(targetAngle);
    // Serial.print("Current Angle: ");
    // Serial.println(all_servo[1].currentAngle);
    // Serial.println();

    delay(20);
  }
}

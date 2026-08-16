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

states original[6]={{0,30},{1,180},{2,0},{3,40},{4,90},{5,0}};
// states original[6]={{0,30},{3,40},{2,0},{1,180},{4,90},{5,0}};
// states original[6]={{0,1},{1,100},{2,90},{3,90},{4,90},{5,0}};


// -------- SERVOS --------
// Servo baseServo;      // rotation
// Servo shoulderServo;  // forward/back
// Servo elbowServo;     // extension
// Servo wristServo;     // pitch
// Servo wristRollServo; // roll (optional)
// Servo gripperServo;   // grab

const int servo_num = 6;
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

Servo all_servo[6] = {{0, 540, 85, 31},{1, 520, 90, 179},{2, 520, 190, 1}, {3, 400, 100, 51}, {4, 550, 100, 91}, {5, 400, 80,1}};
// Servo all_servo[6] = {{0, 540, 85, 0},{1, 520, 90, 111},{2, 520, 190, 91}, {3, 400, 100, 91}, {4, 550, 100, 91}, {5, 400, 80,1}};
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
  Serial.setTxTimeoutMs(0); 


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

int speed=8;
// -------- MOVEMENT --------
void moveLeft(int i)  
{ 
  targetServo=0;
  targetAngle=all_servo[targetServo].currentAngle+i;

  
  angleSet(); 
}
void moveRight(int i)
{
  targetServo=0;
  targetAngle=all_servo[targetServo].currentAngle-i;

  
  angleSet();  
}

void moveForward(int i) 
{ 
  targetServo=1;
  targetAngle=all_servo[targetServo].currentAngle-i;
  angleSet(); 
}
void moveBack(int i)  
{
  targetServo=1;
  targetAngle=all_servo[targetServo].currentAngle+i;
  angleSet();  
}

void extendArm(int i)
{
  targetServo=1;
  targetAngle=all_servo[targetServo].currentAngle+i;
  angleSet(); 
}
void retractArm(int i)  
{ 
  targetServo=2;
  targetAngle=all_servo[targetServo].currentAngle-i;
  angleSet();  
}

void moveDown(int i) {
//adjust elbow=1 and shoulder=2
  targetServo=2;
  targetAngle=all_servo[targetServo].currentAngle+i;
  angleSet();
  targetServo=1;
  targetAngle=all_servo[targetServo].currentAngle-i;
  angleSet();
}

void moveUp(int i) {
  targetServo=2;
  targetAngle=all_servo[targetServo].currentAngle-i;
  angleSet();
  targetServo=1;
  targetAngle=all_servo[targetServo].currentAngle+i;
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
  
  if (Serial.available())
  {
    cmd = Serial.readStringUntil('\n');
    cmd.trim();

    
    
    

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
    else if (state == "ALIGN" && cmd.indexOf(':')>0)
    {
      String direction=cmd.substring(0,cmd.indexOf(':'));
      speed=cmd.substring(cmd.indexOf(':')+1).toInt();

      Serial.print("Command: ");
      Serial.println(direction);
      
      if (direction == "LEFT")
        moveLeft(speed);
      else if (direction == "RIGHT")
        moveRight(speed);
      else if (direction == "FORWARD")
        moveForward(speed);
      else if (direction == "BACK")
        moveBack(speed);

      else if (direction == "CENTERED")
      {
        moveDown(speed);
      }
    }
    else if (cmd=="CATCH")
      {
        state = "GRAB";

      }

    
    // -------- GRAB --------
    if (state == "GRAB")
    {
      close();
      delay(500);

      delay(300);

      state = "DONE";
      speed=8;
    }


    if(cmd=="RESET")
    {
      state="ALIGN";
      for(int i=0;i<6;i++)
      {
        targetServo=original[i].ind;
        targetAngle=original[i].angle;
        angleSet();
      }
    }
    

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

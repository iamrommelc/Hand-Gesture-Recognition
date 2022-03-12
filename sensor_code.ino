int flexSensorPin0 = A0;          //analog pin 0 
int flexSensorPin2 = A1;          //analog pin 1
int flexSensorPin2 = A2;          //analog pin 2

void setup(){
  Serial.begin(9600);
}

void loop()  {
  int flexSensorReading0 = analogRead(flexSensorPin0);
  int flex0 = map(flexSensorReading0, 512, 614,  0, 100);   
  Serial.print(flex0);
  Serial.print(' ');
  
  int flexSensorReading1 = analogRead(flexSensorPin1);
  int flex1 = map(flexSensorReading1, 512, 614,  0, 100);  
  Serial.print(flex1);
  Serial.print(' ');
  
  int flexSensorReading2 = analogRead(flexSensorPin2);
  int flex2 = map(flexSensorReading2, 512, 614,  0, 100);   
  Serial.print(flex2);
  Serial.println(' ');

  delay(250);
}
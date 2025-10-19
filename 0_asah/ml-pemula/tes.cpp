#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x20, 16, 2);

const int buttonPin = 2;
const int redPin = 11;
const int greenPin = 12;
const int bluePin = 13;

bool lampState = false;
int colorState = 0;

void setup()
{
    Serial.begin(9600);
    Wire.begin();
    pinMode(buttonPin, INPUT);
    pinMode(redPin, OUTPUT);
    pinMode(greenPin, OUTPUT);
    pinMode(bluePin, OUTPUT);

    lcd.init();
    lcd.backlight();
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("NIM : 123230099");
    delay(2000);
    showLampOff();
}

void loop()
{
    if (digitalRead(buttonPin) == HIGH)
    {
        lampState = true;
        showLampOn();
    }
    else
    {
        lampState = false;
        showLampOff();
    }

    if (lampState)
    {
        changeColor();
        delay(500);
    }

    if (Serial.available() > 0)
    {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        cmd.toLowerCase();

        if (cmd == "mati")
        {
            lampState = false;
            showLampOff();
        }
        else
        {
            Serial.println("Command Salah");
            lcd.clear();
            lcd.setCursor(0, 0);
            lcd.print("Command Salah");
            delay(1000);
            if (lampState)
                showLampOn();
            else
                showLampOff();
        }
    }
}

void showLampOff()
{
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Lampu Mati");
    analogWrite(redPin, 0);
    analogWrite(greenPin, 0);
    analogWrite(bluePin, 0);
}

void showLampOn()
{
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Lampu Hidup");
}

void changeColor()
{
    switch (colorState)
    {
    case 0:
        analogWrite(redPin, 255);
        analogWrite(greenPin, 0);
        analogWrite(bluePin, 0);
        break;
    case 1:
        analogWrite(redPin, 0);
        analogWrite(greenPin, 255);
        analogWrite(bluePin, 0);
        break;
    case 2:
        analogWrite(redPin, 0);
        analogWrite(greenPin, 0);
        analogWrite(bluePin, 255);
        break;
    }
    colorState = (colorState + 1) % 3;
}

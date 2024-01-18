import RPi.GPIO as GPIO
import time

LED_PIN = 17
LED_PIN_2 = 27
LED_PIN_3 = 22
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(LED_PIN_2, GPIO.OUT)
GPIO.setup(LED_PIN_3, GPIO.OUT)


GPIO.output(LED_PIN_2, GPIO.LOW)
GPIO.output(LED_PIN, GPIO.LOW)
count = 0
# while True:
#     time.sleep(3)

#     GPIO.output(LED_PIN, GPIO.HIGH)
#     GPIO.output(LED_PIN_2, GPIO.HIGH)
#     GPIO.output(LED_PIN_3, GPIO.HIGH)

#     time.sleep(0.5)

#     GPIO.output(LED_PIN, GPIO.LOW)
#     GPIO.output(LED_PIN_2, GPIO.LOW)
#     GPIO.output(LED_PIN_3, GPIO.LOW)              
#     if count == 3:
#         break
#     count+=1

GPIO.cleanup()
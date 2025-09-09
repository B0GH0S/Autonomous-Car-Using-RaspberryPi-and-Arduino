import serial
import time

class Motor:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        # Initialize serial communication with the Arduino
        self.ser = serial.Serial(port, baudrate)
        time.sleep(2)  # Wait for the serial connection to establish

    def move(self, speed=1, turn=0, t=0):
    # If turn value is significant enough, send turn command
        if abs(turn) > 0.1:  # Add threshold for turn
            if turn > 0:
                self.ser.write('a'.encode())
            else:
                self.ser.write('d'.encode())
        # If no significant turn, use forward/backward based on speed
        else:
            if speed > 0:
                self.ser.write('w'.encode())
            elif speed < 0:
                self.ser.write('s'.encode())
            else:
                self.ser.write('x'.encode())
     

    def stop(self, t=0):
            # Command to stop the motors
            self.ser.write('x'.encode())
            time.sleep(t)


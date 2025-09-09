# Autonomous-Car-Using-RaspberryPi-and-Arduino
This repository is the result of piecing together an autonomous vehicle from sparse information and incompatible libraries. If you're trying to get a Raspberry Pi (for a brain) to talk to an Arduino (for muscle control) to do something intelligent, this code might save you weeks of headache.

It includes:

The core Python/OpenCV logic for basic lane detection.

The Arduino C++ code for handling a motor driver and 4 motors

A working Serial communication protocol between the two boards.

I'm not sure of the exact library versions I used, but the core concepts are here. Use this as a starting point to understand the architecture, then build your own amazing project on top of it.

Materials Used:
  Raspberry Pi 4 (Recommended to get a fan for it as well)
  Arduino Uno
  Arduino Uno Motor Bridge
  64GB Micro SD Card
  Pi Camera 2
  Plastic Car Chassis
  4 DC Motors
  4 Wheels

# Vine-Robot
╔══════════════════════════════════════════════════════════════╗
║         PREFORMED VINE ROBOT — INTEGRATED SYSTEM            ║
║                                                              ║
║  Webcam (YOLOv8)  →  Python Brain  →  Arduino + L298N       ║
║                             ↕                               ║
║                    PyBullet 3D Simulation                    ║
╚══════════════════════════════════════════════════════════════╝

Requirements:
    pip install ultralytics opencv-python pyserial pybullet

Hardware:
    - Laptop webcam
    - Arduino Uno connected via USB
    - L298N Motor Driver + DC Motor
    - Arduino must be running PreformedVine_KeyboardControl.ino

Usage:
    1. Update SERIAL_PORT below to match your Arduino
    2. Run: python vine_robot_integrated.py
    3. Press Q in the webcam window to quit

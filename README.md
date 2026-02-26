# Vine-Robot
LAPTOP WEBCAM AND VINE ROBOT SYSTEM WITH THERMAL HUMAN DETECTION  
Webcam → Thermal Filter → YOLOv8 Nano → Arduino + L298N → PyBullet                    

Press T → Toggle between Thermal and Normal View
Press Q → Quit

Requirements:  
pip install ultralytics opencv-python pyserial pybullet

Hardware:  
Laptop webcam  
Arduino Uno connected via USB  
L298N Motor Driver + DC Motor  
Arduino must be running PreformedVine_KeyboardControl.ino

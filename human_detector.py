"""
Human Detection → Arduino Vine Stop
====================================
Uses your laptop webcam + YOLOv8 to detect humans.
When a human is detected, sends a STOP command ('space' key) to the Arduino.

Requirements:
    pip install ultralytics opencv-python pyserial

Usage:
    1. Find your Arduino's COM port:
       - Windows: Check Device Manager → Ports (COM & LPT)
       - Mac/Linux: ls /dev/tty.*
    2. Update SERIAL_PORT below
    3. Run: python human_detector.py
    4. Press Q to quit
"""

import cv2
import serial
import time
from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIGURATION — edit these to match your setup
# ─────────────────────────────────────────
SERIAL_PORT     = "COM7"       # Windows: "COM7", Mac/Linux: "/dev/ttyUSB0" or "/dev/tty.usbmodem..."
BAUD_RATE       = 9600         # Must match Serial.begin() in your Arduino code
CONFIDENCE      = 0.5          # Detection confidence threshold (0.0 – 1.0)
CAMERA_INDEX    = 0            # 0 = default laptop webcam
COOLDOWN_SEC    = 2.0          # Seconds to wait before sending another stop command
# ─────────────────────────────────────────

def main():
    # Load YOLOv8 nano (smallest/fastest model, downloads automatically ~6MB)
    print("[INFO] Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    # Open serial connection to Arduino
    print(f"[INFO] Connecting to Arduino on {SERIAL_PORT}...")
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset after serial connection
        print("[INFO] Arduino connected!")
    except Exception as e:
        print(f"[ERROR] Could not connect to Arduino: {e}")
        print("[TIP] Check your SERIAL_PORT setting or make sure Arduino is plugged in.")
        arduino = None

    # Open webcam
    print("[INFO] Opening webcam...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    last_stop_time = 0
    print("[INFO] Watching for humans... Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 detection
        results = model(frame, verbose=False)[0]

        human_detected = False

        for box in results.boxes:
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            label = model.names[cls]

            # Class 0 in COCO dataset = "person"
            if label == "person" and conf >= CONFIDENCE:
                human_detected = True

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Human {conf:.0%}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Send STOP command if human detected and cooldown has passed
        now = time.time()
        if human_detected and (now - last_stop_time) > COOLDOWN_SEC:
            print("[ACTION] Human detected! Sending STOP to Arduino...")

            if arduino and arduino.is_open:
                arduino.write(b'X')   # Space = halt(X) in your Arduino code
                print("[SENT] STOP command sent via serial.\n")
            else:
                print("[SIMULATED] Would send STOP (no Arduino connected).\n")

            last_stop_time = now

        # Display status on frame
        status = "HUMAN DETECTED - STOPPED" if human_detected else "Watching..."
        color  = (0, 0, 255) if human_detected else (255, 255, 255)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Vine Robot - Human Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if arduino and arduino.is_open:
        arduino.close()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    main()

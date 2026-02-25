import cv2
import serial
import time
import threading
import math
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION — Edit these to match your setup
# ─────────────────────────────────────────────────────────────
SERIAL_PORT    = "COM7"        # Windows: "COM7" | Mac/Linux: "/dev/ttyUSB0"
BAUD_RATE      = 9600          # Must match Arduino Serial.begin()
CONFIDENCE     = 0.5           # YOLO detection confidence (0.0–1.0)
CAMERA_INDEX   = 0             # 0 = default laptop webcam
COOLDOWN_SEC   = 1.5           # Seconds between Arduino commands
USE_SIMULATION = True          # Set False if PyBullet causes issues
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
#  3D SIMULATION MODULE (PyBullet)
# ══════════════════════════════════════════════════════════════
class VineSimulation:
    """
    PyBullet 3D simulation of the vine robot.
    - Blue cylinder  = the vine body
    - Red box        = detected human
    - Green platform = ground/floor
    """

    def __init__(self):
        self.running   = False
        self.state     = "IDLE"      # GROWING | RETRACTING | STOPPED | IDLE
        self.vine_len  = 0.3         # Current vine length in meters
        self.min_len   = 0.1         # Fully retracted
        self.max_len   = 1.5         # Fully extended
        self.grow_rate = 0.008       # Meters added per sim step
        self.human_id  = None
        self.vine_id   = None
        self.p         = None        # PyBullet physics client

    def start(self):
        """Initialize PyBullet and build the environment."""
        try:
            import pybullet as pb
            import pybullet_data
            self.p = pb
            self.client = pb.connect(pb.GUI)

            # Window title & camera angle
            pb.resetDebugVisualizerCamera(
                cameraDistance=4.0,
                cameraYaw=180,       # Look along X axis from the side
                cameraPitch=-20,
                cameraTargetPosition=[0.75, 0, 0.3]
            )
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            pb.setGravity(0, 0, -9.8)

            # ── Ground plane ──────────────────────────────────
            pb.loadURDF("plane.urdf")

            # ── Vine base platform (gray box) ─────────────────
            base_col  = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.15, 0.15, 0.05])
            base_vis  = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.15, 0.15, 0.05],
                                              rgbaColor=[0.4, 0.4, 0.4, 1])
            pb.createMultiBody(0, base_col, base_vis, [0, 0, 0.05])

            # ── Vine body (blue cylinder, starts short) ───────
            self.vine_id = self._make_vine(self.vine_len)

            # ── Human placeholder (red box, hidden at start) ──
            self.human_id = self._make_human(visible=False)

            # ── Status text ───────────────────────────────────
            self.text_id = pb.addUserDebugText(
                "STATUS: IDLE",
                textPosition=[0.5, -1.5, 1.2],
                textColorRGB=[1, 1, 1],
                textSize=1.5
            )

            self.running = True
            print("[SIM] PyBullet simulation started ✓")

        except ImportError:
            print("[SIM] PyBullet not found. Simulation disabled.")
            print("[SIM] Install with: pip install pybullet")
            self.running = False
        except Exception as e:
            print(f"[SIM] Could not start simulation: {e}")
            self.running = False

    def _make_vine(self, length):
        """Create/replace the vine cylinder growing along the X axis."""
        pb = self.p
        if self.vine_id is not None:
            pb.removeBody(self.vine_id)

        radius   = 0.04
        half_len = length / 2

        # Rotate 90 degrees around Y axis so cylinder lies flat along X
        orn = pb.getQuaternionFromEuler([0, math.pi / 2, 0])

        col = pb.createCollisionShape(pb.GEOM_CYLINDER,
                                       radius=radius, height=length)
        vis = pb.createVisualShape(pb.GEOM_CYLINDER,
                                    radius=radius, length=length,
                                    rgbaColor=[0.2, 0.5, 1.0, 1.0])  # Blue

        # Center of cylinder sits at its midpoint along X; raised slightly off ground
        vine_id = pb.createMultiBody(0, col, vis,
                                      basePosition=[half_len, 0, 0.15],
                                      baseOrientation=orn)
        return vine_id

    def _make_human(self, visible=True):
        """Create a red box in front of the vine tip along the X axis."""
        pb = self.p
        if self.human_id is not None:
            pb.removeBody(self.human_id)

        alpha = 1.0 if visible else 0.0
        # Human box dimensions: shallow in X (depth), wide in Y, tall in Z
        col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.1, 0.25, 0.45])
        vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.1, 0.25, 0.45],
                                    rgbaColor=[1.0, 0.1, 0.1, alpha])  # Red

        # Position just ahead of the vine tip on the X axis
        tip_x    = self.vine_len          # Current vine tip position
        human_id = pb.createMultiBody(0, col, vis,
                                       basePosition=[tip_x + 0.35, 0, 0.5])
        return human_id

    def set_state(self, new_state):
        """Called from main thread to update vine behavior."""
        self.state = new_state

    def update(self):
        """
        Step the simulation forward — call this every frame.
        Grows or retracts the vine, shows/hides human.
        """
        if not self.running:
            return
        pb = self.p

        try:
            # ── Update vine length based on state ─────────────
            if self.state == "GROWING":
                self.vine_len = min(self.vine_len + self.grow_rate, self.max_len)
                self.vine_id  = self._make_vine(self.vine_len)
                label_color   = [0.2, 1.0, 0.2]   # Green
                label_text    = f"STATUS: GROWING  [{self.vine_len:.2f}m]"

            elif self.state == "RETRACTING":
                self.vine_len = max(self.vine_len - self.grow_rate, self.min_len)
                self.vine_id  = self._make_vine(self.vine_len)
                label_color   = [1.0, 0.3, 0.3]   # Red
                label_text    = f"STATUS: RETRACTING  [{self.vine_len:.2f}m]"

            else:  # STOPPED / IDLE
                label_color = [1.0, 1.0, 0.0]      # Yellow
                label_text  = f"STATUS: STOPPED  [{self.vine_len:.2f}m]"

            # ── Show/hide human box ───────────────────────────
            human_visible = self.state == "RETRACTING"
            self.human_id = self._make_human(visible=human_visible)

            # ── Update status text ────────────────────────────
            pb.removeUserDebugItem(self.text_id)
            self.text_id = pb.addUserDebugText(
                label_text,
                textPosition=[0.5, -1.5, 1.2],
                textColorRGB=label_color,
                textSize=1.5
            )

            pb.stepSimulation()

        except Exception as e:
            print(f"[SIM] Update error: {e}")
            self.running = False

    def stop(self):
        if self.running and self.p:
            try:
                self.p.disconnect()
            except:
                pass
        self.running = False


# ══════════════════════════════════════════════════════════════
#  ARDUINO COMMUNICATION
# ══════════════════════════════════════════════════════════════
class ArduinoController:
    def __init__(self, port, baud):
        self.port   = port
        self.baud   = baud
        self.serial = None
        self._connect()

    def _connect(self):
        print(f"[ARDUINO] Connecting on {self.port}...")
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)   # Let Arduino reset
            print("[ARDUINO] Connected ✓")
        except Exception as e:
            print(f"[ARDUINO] Connection failed: {e}")
            print("[ARDUINO] Running in simulation-only mode.\n")
            self.serial = None

    def send(self, command: bytes, label=""):
        if self.serial and self.serial.is_open:
            self.serial.write(command)
            print(f"[ARDUINO] Sent '{command.decode()}' → {label}")
        else:
            print(f"[SIMULATED] Would send '{command.decode()}' → {label}")

    def grow(self):
        self.send(b'W', "GROW")

    def stop(self):
        self.send(b'X', "STOP/RETRACT")

    def close(self):
        if self.serial and self.serial.is_open:
            self.serial.close()


# ══════════════════════════════════════════════════════════════
#  MAIN INTEGRATION LOOP
# ══════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*60)
    print("  VINE ROBOT — INTEGRATED CONTROL SYSTEM")
    print("="*60)
    print("  Webcam + YOLOv8 + PyBullet + Arduino")
    print("  Press Q in the webcam window to quit")
    print("="*60 + "\n")

    # ── Load YOLO model ──────────────────────────────────────
    print("[YOLO] Loading YOLOv8 model (downloads ~6MB on first run)...")
    model = YOLO("yolov8n.pt")
    print("[YOLO] Model loaded ✓")

    # ── Open webcam ─────────────────────────────────────────
    print(f"[CAM] Opening webcam (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[CAM] ERROR: Could not open webcam.")
        return
    print("[CAM] Webcam opened ✓")

    # ── Initialize Arduino ──────────────────────────────────
    arduino = ArduinoController(SERIAL_PORT, BAUD_RATE)

    # ── Initialize 3D Simulation ────────────────────────────
    sim = VineSimulation()
    if USE_SIMULATION:
        sim.start()

    # ── State tracking ──────────────────────────────────────
    last_command_time = 0
    current_state     = "GROWING"   # Start by growing

    print("\n[SYSTEM] All systems ready. Watching for humans...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CAM] Frame read failed.")
                break

            # ── Run YOLO detection ───────────────────────────
            results        = model(frame, verbose=False)[0]
            human_detected = False
            human_count    = 0

            for box in results.boxes:
                cls   = int(box.cls[0])
                conf  = float(box.conf[0])
                label = model.names[cls]

                if label == "person" and conf >= CONFIDENCE:
                    human_detected = True
                    human_count   += 1

                    # Draw bounding box on webcam feed
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"HUMAN {conf:.0%}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            now = time.time()

            # ── Decision logic & command sending ─────────────
            if human_detected:
                if current_state != "RETRACTING" and (now - last_command_time) > COOLDOWN_SEC:
                    print(f"[DETECT] 🔴 Human(s) detected ({human_count})! → RETRACTING")
                    arduino.stop()
                    current_state     = "RETRACTING"
                    last_command_time = now

            else:  # No human
                if current_state != "GROWING" and (now - last_command_time) > COOLDOWN_SEC:
                    print("[DETECT] 🟢 Path clear → GROWING")
                    arduino.grow()
                    current_state     = "GROWING"
                    last_command_time = now

            # ── Update 3D simulation ─────────────────────────
            sim.set_state(current_state)
            sim.update()

            # ── Overlay status on webcam feed ─────────────────
            if human_detected:
                status_text  = f"⚠  HUMAN DETECTED ({human_count}) — RETRACTING"
                status_color = (0, 0, 255)    # Red
                bg_color     = (0, 0, 120)
            else:
                status_text  = "✓  CLEAR — GROWING"
                status_color = (0, 255, 80)   # Green
                bg_color     = (0, 80, 0)

            # Status bar background
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), bg_color, -1)
            cv2.putText(frame, status_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

            # Vine length indicator
            vine_pct  = (sim.vine_len - sim.min_len) / (sim.max_len - sim.min_len)
            bar_width = int(vine_pct * (frame.shape[1] - 20))
            cv2.rectangle(frame, (10, frame.shape[0]-30),
                          (10 + bar_width, frame.shape[0]-10), (0, 180, 255), -1)
            cv2.putText(frame, f"Vine: {sim.vine_len:.2f}m",
                        (10, frame.shape[0]-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # ── Show webcam window ────────────────────────────
            cv2.imshow("Vine Robot — Human Detector (Q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[SYSTEM] Q pressed — shutting down...")
                break

    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted.")

    finally:
        # ── Cleanup ──────────────────────────────────────────
        print("[SYSTEM] Cleaning up...")
        arduino.stop()          # Safety: stop motor before exit
        time.sleep(0.5)
        cap.release()
        cv2.destroyAllWindows()
        arduino.close()
        sim.stop()
        print("[SYSTEM] Shutdown complete. ✓")


if __name__ == "__main__":
    main()

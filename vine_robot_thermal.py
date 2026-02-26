import cv2
import serial
import time
import math
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────
SERIAL_PORT    = "COM7"    # Windows: "COM7" | Mac/Linux: "/dev/ttyUSB0"
BAUD_RATE      = 9600
CONFIDENCE     = 0.5      # Slightly lower threshold works well on thermal
CAMERA_INDEX   = 0
COOLDOWN_SEC   = 1.5
USE_SIMULATION = True      # Set False if PyBullet causes issues

# Thermal palette options:
#   cv2.COLORMAP_INFERNO  → most realistic SAR thermal (recommended)
#   cv2.COLORMAP_JET      → classic blue-green-red heatmap
#   cv2.COLORMAP_HOT      → black → red → yellow → white
THERMAL_PALETTE = cv2.COLORMAP_INFERNO
# ─────────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════════
#  THERMAL FILTER
# ═════════════════════════════════════════════════════════════════
def apply_thermal_filter(frame):
    """
    Converts a normal BGR webcam frame into a thermal-style image.

    Pipeline:
      BGR  →  Grayscale  →  CLAHE (local contrast)  →  Inferno colormap

    CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances
    local brightness differences so that a human body (slightly warmer /
    brighter under normal light) appears as a hotter region in the output.
    """
    # Step 1 — Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2 — CLAHE: boosts contrast locally (8x8 tiles, clip=3.0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 3 — Apply thermal colormap
    thermal = cv2.applyColorMap(enhanced, THERMAL_PALETTE)

    return thermal


def add_thermal_overlay(thermal_frame, frame_width, frame_height):
    """Add a subtle scanline texture and corner brackets to sell the effect."""

    # Scanlines (every 4 rows, slightly darken)
    overlay = thermal_frame.copy()
    for y in range(0, frame_height, 4):
        overlay[y, :] = (overlay[y, :] * 0.6).astype(np.uint8)

    # Corner brackets (top-left, top-right, bottom-left, bottom-right)
    bracket_color = (180, 180, 180)
    blen = 20  # bracket arm length
    bthk = 2   # thickness
    corners = [
        ((10, 10), (10 + blen, 10), (10, 10 + blen)),           # top-left
        ((frame_width-10, 10), (frame_width-10-blen, 10),
         (frame_width-10, 10+blen)),                             # top-right
        ((10, frame_height-10), (10+blen, frame_height-10),
         (10, frame_height-10-blen)),                            # bottom-left
        ((frame_width-10, frame_height-10),
         (frame_width-10-blen, frame_height-10),
         (frame_width-10, frame_height-10-blen)),                # bottom-right
    ]
    for corner, h_end, v_end in corners:
        cv2.line(overlay, corner, h_end, bracket_color, bthk)
        cv2.line(overlay, corner, v_end, bracket_color, bthk)

    # Blend scanline overlay
    result = cv2.addWeighted(thermal_frame, 0.7, overlay, 0.3, 0)
    return result


# ═════════════════════════════════════════════════════════════════
#  3D SIMULATION (PyBullet)
# ═════════════════════════════════════════════════════════════════
class VineSimulation:
    def __init__(self):
        self.running   = False
        self.state     = "IDLE"
        self.vine_len  = 0.3
        self.min_len   = 0.1
        self.max_len   = 1.5
        self.grow_rate = 0.008
        self.human_id  = None
        self.vine_id   = None
        self.text_id   = None
        self.p         = None

    def start(self):
        try:
            import pybullet as pb
            import pybullet_data
            self.p = pb
            pb.connect(pb.GUI)
            pb.resetDebugVisualizerCamera(
                cameraDistance=4.0,
                cameraYaw=180,
                cameraPitch=-20,
                cameraTargetPosition=[0.75, 0, 0.3]
            )
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            pb.setGravity(0, 0, -9.8)
            pb.loadURDF("plane.urdf")

            # Gray base platform
            bc = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.15, 0.15, 0.05])
            bv = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.15, 0.15, 0.05],
                                       rgbaColor=[0.4, 0.4, 0.4, 1])
            pb.createMultiBody(0, bc, bv, [0, 0, 0.05])

            self.vine_id  = self._make_vine(self.vine_len)
            self.human_id = self._make_human(visible=False)
            self.text_id  = pb.addUserDebugText(
                "STATUS: IDLE", [0.5, -1.5, 1.2],
                textColorRGB=[1, 1, 1], textSize=1.5
            )
            self.running = True
            print("[SIM] PyBullet started ✓")
        except ImportError:
            print("[SIM] PyBullet not installed — run: pip install pybullet")
            self.running = False
        except Exception as e:
            print(f"[SIM] Failed to start: {e}")
            self.running = False

    def _make_vine(self, length):
        pb = self.p
        if self.vine_id is not None:
            pb.removeBody(self.vine_id)
        half  = length / 2
        orn   = pb.getQuaternionFromEuler([0, math.pi / 2, 0])
        col   = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=0.04, height=length)
        vis   = pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.04, length=length,
                                      rgbaColor=[0.2, 0.5, 1.0, 1.0])
        return pb.createMultiBody(0, col, vis,
                                   basePosition=[half, 0, 0.15],
                                   baseOrientation=orn)

    def _make_human(self, visible=True):
        pb = self.p
        if self.human_id is not None:
            pb.removeBody(self.human_id)
        alpha = 1.0 if visible else 0.0
        col   = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.1, 0.25, 0.45])
        vis   = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.1, 0.25, 0.45],
                                      rgbaColor=[1.0, 0.1, 0.1, alpha])
        return pb.createMultiBody(0, col, vis,
                                   basePosition=[self.vine_len + 0.35, 0, 0.5])

    def set_state(self, new_state):
        self.state = new_state

    def update(self):
        if not self.running:
            return
        pb = self.p
        try:
            if self.state == "GROWING":
                self.vine_len = min(self.vine_len + self.grow_rate, self.max_len)
                self.vine_id  = self._make_vine(self.vine_len)
                color = [0.2, 1.0, 0.2]
                text  = f"STATUS: GROWING  [{self.vine_len:.2f}m]"
            elif self.state == "RETRACTING":
                self.vine_len = max(self.vine_len - self.grow_rate, self.min_len)
                self.vine_id  = self._make_vine(self.vine_len)
                color = [1.0, 0.3, 0.3]
                text  = f"STATUS: RETRACTING  [{self.vine_len:.2f}m]"
            else:
                color = [1.0, 1.0, 0.0]
                text  = f"STATUS: STOPPED  [{self.vine_len:.2f}m]"

            self.human_id = self._make_human(visible=(self.state == "RETRACTING"))
            pb.removeUserDebugItem(self.text_id)
            self.text_id = pb.addUserDebugText(
                text, [0.5, -1.5, 1.2], textColorRGB=color, textSize=1.5
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


# ═════════════════════════════════════════════════════════════════
#  ARDUINO CONTROLLER
# ═════════════════════════════════════════════════════════════════
class ArduinoController:
    def __init__(self, port, baud):
        self.serial = None
        print(f"[ARDUINO] Connecting on {port}...")
        try:
            self.serial = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            print("[ARDUINO] Connected ✓")
        except Exception as e:
            print(f"[ARDUINO] Not connected ({e}) — simulation-only mode.")

    def send(self, cmd, label):
        if self.serial and self.serial.is_open:
            self.serial.write(cmd)
        print(f"[ARDUINO] {'Sent' if self.serial else 'Simulated'} '{cmd.decode()}' → {label}")

    def grow(self):   self.send(b'W', "GROW")
    def stop(self):   self.send(b'X', "STOP/RETRACT")
    def close(self):
        if self.serial and self.serial.is_open:
            self.serial.close()


# ═════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*62)
    print("  VINE ROBOT — THERMAL VISION + HUMAN DETECTION SYSTEM")
    print("="*62)
    print("  T = Toggle thermal / normal view")
    print("  Q = Quit")
    print("="*62 + "\n")

    # Load model
    print("[YOLO] Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("[YOLO] Ready ✓")

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[CAM] ERROR: Cannot open webcam.")
        return
    print("[CAM] Webcam opened ✓")

    # Arduino + sim
    arduino = ArduinoController(SERIAL_PORT, BAUD_RATE)
    sim = VineSimulation()
    if USE_SIMULATION:
        sim.start()

    # State
    thermal_mode      = True   # Start in thermal view
    last_command_time = 0
    current_state     = "GROWING"

    print("\n[SYSTEM] Running. Press T to toggle thermal view.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # ── Run YOLO on the raw frame (better accuracy) ───────
            results        = model(frame, verbose=False)[0]
            human_detected = False
            human_count    = 0
            detections     = []   # store boxes to draw later

            for box in results.boxes:
                cls   = int(box.cls[0])
                conf  = float(box.conf[0])
                label = model.names[cls]
                if label == "person" and conf >= CONFIDENCE:
                    human_detected = True
                    human_count   += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2, conf))

            # ── Build display frame ───────────────────────────────
            if thermal_mode:
                display = apply_thermal_filter(frame)
                display = add_thermal_overlay(display, w, h)
                # Mode watermark
                cv2.putText(display, "THERMAL", (w - 105, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1)
            else:
                display = frame.copy()
                cv2.putText(display, "NORMAL", (w - 90, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # ── Draw detection boxes on display frame ─────────────
            for (x1, y1, x2, y2, conf) in detections:
                # Cyan boxes stand out well on both thermal and normal views
                box_color  = (0, 255, 255)
                glow_color = (0, 120, 120)

                # Glow effect (draw slightly larger box behind)
                cv2.rectangle(display, (x1-2, y1-2), (x2+2, y2+2), glow_color, 4)
                cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)

                # Label badge
                label_text = f"HUMAN  {conf:.0%}"
                (lw, lh), _ = cv2.getTextSize(label_text,
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(display, (x1, y1 - lh - 10), (x1 + lw + 6, y1),
                              (0, 0, 0), -1)
                cv2.putText(display, label_text, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)

                # Heat indicator bar (simulates thermal intensity)
                bar_h = y2 - y1
                bar_x = x2 + 6
                if bar_x + 12 < w:
                    for i in range(bar_h):
                        # Gradient: blue (cold) → white (hot) from bottom to top
                        ratio = 1.0 - (i / bar_h)
                        r = int(255 * ratio)
                        g = int(100 * ratio)
                        b = int(255 * (1 - ratio))
                        cv2.line(display,
                                 (bar_x, y1 + i), (bar_x + 8, y1 + i),
                                 (b, g, r), 1)

            # ── Arduino + sim state decisions ─────────────────────
            now = time.time()
            if human_detected:
                if current_state != "RETRACTING" and \
                        (now - last_command_time) > COOLDOWN_SEC:
                    print(f"[DETECT] 🔴 Human(s) detected ({human_count}) → RETRACTING")
                    arduino.stop()
                    current_state     = "RETRACTING"
                    last_command_time = now
            else:
                if current_state != "GROWING" and \
                        (now - last_command_time) > COOLDOWN_SEC:
                    print("[DETECT] 🟢 Clear → GROWING")
                    arduino.grow()
                    current_state     = "GROWING"
                    last_command_time = now

            sim.set_state(current_state)
            sim.update()

            # ── Status bar (top) ──────────────────────────────────
            if human_detected:
                status = f"[!] HUMAN DETECTED ({human_count})  -  RETRACTING"
                bar_bg   = (0, 0, 140)
                bar_text = (0, 255, 255)
            else:
                status = "[OK] CLEAR  -  GROWING"
                bar_bg   = (0, 80, 0)
                bar_text = (0, 255, 80)

            cv2.rectangle(display, (0, 0), (w, 48), bar_bg, -1)
            cv2.putText(display, status, (10, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, bar_text, 2)

            # Thermal/normal toggle hint
            cv2.putText(display, "[T] Toggle View  [Q] Quit",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (160, 160, 160), 1)

            # ── Vine length bar (bottom) ──────────────────────────
            vine_pct  = (sim.vine_len - sim.min_len) / (sim.max_len - sim.min_len)
            bar_w     = int(vine_pct * (w - 20))
            cv2.rectangle(display, (10, h-32), (10 + bar_w, h-20),
                          (0, 200, 255), -1)
            cv2.putText(display, f"Vine: {sim.vine_len:.2f}m",
                        (10, h - 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1)

            cv2.imshow("Vine Robot — Thermal Vision  [T=toggle  Q=quit]", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[SYSTEM] Quitting...")
                break
            elif key == ord('t') or key == ord('T'):
                thermal_mode = not thermal_mode
                mode_str = "THERMAL" if thermal_mode else "NORMAL RGB"
                print(f"[VIEW] Switched to {mode_str} mode")

    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted.")

    finally:
        print("[SYSTEM] Shutting down...")
        arduino.stop()
        time.sleep(0.5)
        cap.release()
        cv2.destroyAllWindows()
        arduino.close()
        sim.stop()
        print("[SYSTEM] Done ✓")


if __name__ == "__main__":
    main()

import cv2
import mediapipe as mp
import numpy as np
import wave
import os
import pygame
import time
import serial
import serial.tools.list_ports

# ─────────────────────────────────────────
# ESP32 CONNECTION (optional)
# ─────────────────────────────────────────
def find_esp32():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = p.description.upper()
        if any(x in desc for x in ["CP210", "CH340", "CH341", "UART", "USB SERIAL"]):
            return p.device
    return None

esp32 = None
port  = find_esp32()
if port:
    try:
        esp32 = serial.Serial(port, 9600, timeout=1)
        time.sleep(1.5)
        print(f"ESP32 connected on {port}")
    except:
        print("ESP32 found but could not open. Continuing without buzzer and LED.")
else:
    print("ESP32 not detected. Running laptop-only mode.")

SOUND_MODE = "both" if esp32 else "laptop"

# ─────────────────────────────────────────
# LED MAPPING (8 notes → 4 LEDs)
# ─────────────────────────────────────────
def get_led_for_note(note_index):
    """Maps 8 notes to 4 LEDs (pairs of 2)"""
    return note_index // 2  # 0-7 → 0-3

# ─────────────────────────────────────────
# GENERATE PIANO NOTE WAV FILES
# ─────────────────────────────────────────
def generate_piano_note(filename, frequency, duration=1.5, sample_rate=44100):
    if os.path.exists(filename):
        return
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave_data = (
        1.0  * np.sin(2 * np.pi * frequency * 1 * t) +
        0.5  * np.sin(2 * np.pi * frequency * 2 * t) +
        0.25 * np.sin(2 * np.pi * frequency * 3 * t) +
        0.12 * np.sin(2 * np.pi * frequency * 4 * t) +
        0.06 * np.sin(2 * np.pi * frequency * 5 * t)
    )
    envelope  = np.exp(-3 * t)
    wave_data = wave_data * envelope
    wave_data = wave_data / np.max(np.abs(wave_data))
    wave_data = (wave_data * 32767).astype(np.int16)
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(wave_data.tobytes())

FREQUENCIES = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
NOTE_FILES  = [f"note_{i}.wav" for i in range(8)]

print("Generating piano notes...")
for i, freq in enumerate(FREQUENCIES):
    generate_piano_note(NOTE_FILES[i], freq)
print("Done!")

# ─────────────────────────────────────────
# PYGAME INIT
# ─────────────────────────────────────────
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
pygame.mixer.set_num_channels(8)
sounds   = [pygame.mixer.Sound(f) for f in NOTE_FILES]
channels = [pygame.mixer.Channel(i) for i in range(8)]

# ─────────────────────────────────────────
# MEDIAPIPE INIT
# ─────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ─────────────────────────────────────────
# WEBCAM + KEY LAYOUT
# ─────────────────────────────────────────
cap = cv2.VideoCapture(0)
ret, test_frame = cap.read()
FRAME_H, FRAME_W = test_frame.shape[:2]

NUM_KEYS    = 8
GAP         = 6
KEY_H       = 160
KEY_W       = (FRAME_W - GAP * (NUM_KEYS + 1)) // NUM_KEYS
KEY_Y       = FRAME_H - KEY_H - 10
total_width = NUM_KEYS * KEY_W + (NUM_KEYS - 1) * GAP
START_X     = (FRAME_W - total_width) // 2

NOTE_NAMES = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C2']

keys = []
for i in range(NUM_KEYS):
    x1 = START_X + i * (KEY_W + GAP)
    x2 = x1 + KEY_W
    keys.append((x1, KEY_Y, x2, KEY_Y + KEY_H))

# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
last_pressed      = -1
currently_playing = -1
last_press_time   = 0
COOLDOWN          = 0.4

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    finger_x, finger_y = -1, -1

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            tip      = hand_landmarks.landmark[8]
            h, w, _  = frame.shape
            finger_x = int(tip.x * w)
            finger_y = int(tip.y * h)
            cv2.circle(frame, (finger_x, finger_y), 12, (0, 255, 0), -1)

    # ── draw keys + detect press ──
    pressed_key = -1
    for i, (x1, y1, x2, y2) in enumerate(keys):
        touching = x1 < finger_x < x2 and y1 < finger_y < y2
        color    = (0, 255, 255) if touching else (255, 255, 255)
        if touching:
            pressed_key = i

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, NOTE_NAMES[i],
                    (x1 + KEY_W // 2 - 10, y2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── sound + buzzer + LED logic ──
    now = time.time()

    if pressed_key == -1:
        last_pressed = -1

    elif pressed_key != last_pressed:
        if currently_playing != -1:
            channels[currently_playing].stop()

        if SOUND_MODE in ("laptop", "both"):
            channels[pressed_key].play(sounds[pressed_key])
            currently_playing = pressed_key

        if SOUND_MODE in ("buzzer", "both") and esp32:
            try:
                led_num = get_led_for_note(pressed_key)
                esp32.write(f"NOTE:{pressed_key},LED:{led_num}\n".encode())
            except:
                pass

        last_pressed    = pressed_key
        last_press_time = now

    elif (now - last_press_time) > COOLDOWN:
        if SOUND_MODE in ("laptop", "both"):
            channels[pressed_key].stop()
            channels[pressed_key].play(sounds[pressed_key])

        if SOUND_MODE in ("buzzer", "both") and esp32:
            try:
                led_num = get_led_for_note(pressed_key)
                esp32.write(f"NOTE:{pressed_key},LED:{led_num}\n".encode())
            except:
                pass

        last_press_time = now

    # ── UI ──
    mode_colors = {
        "laptop": (100, 200, 255),
        "buzzer": (100, 255, 150),
        "both":   (255, 200, 50)
    }
    cv2.putText(frame, "AI Virtual Piano", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
    cv2.putText(frame, "Point index finger at a key", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, f"MODE: {SOUND_MODE.upper()}   (1=Laptop  2=Buzzer  3=Both)",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                mode_colors[SOUND_MODE], 2)
    if not esp32:
        cv2.putText(frame, "ESP32 not connected", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)

    cv2.imshow("AI Virtual Piano", frame)

    key_pressed = cv2.waitKey(1)
    if key_pressed == ord('1'):
        SOUND_MODE = "laptop"
        print("Mode: Laptop only")
    elif key_pressed == ord('2'):
        if esp32:
            SOUND_MODE = "buzzer"
            print("Mode: Buzzer only")
        else:
            print("ESP32 not connected — staying on laptop mode")
    elif key_pressed == ord('3'):
        if esp32:
            SOUND_MODE = "both"
            print("Mode: Both")
        else:
            print("ESP32 not connected — staying on laptop mode")
    elif key_pressed == 27:
        break

cap.release()
cv2.destroyAllWindows()
if esp32:
    esp32.close()

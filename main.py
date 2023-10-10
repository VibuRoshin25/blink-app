import cv2
import dlib
import math
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, font
from PIL import Image, ImageTk

BLINK_RATIO_THRESHOLD = 4
BLINK_DURATION = 60  # Duration in seconds
COOLDOWN_TIME = 1.5  # Cooldown time in seconds
MIN_BLINK_COUNT = 15

def midpoint(point1, point2):
    return (point1.x + point2.x) / 2, (point1.y + point2.y) / 2

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def get_blink_ratio(eye_points, facial_landmarks):
    corner_left = (
        facial_landmarks.part(eye_points[0]).x,
        facial_landmarks.part(eye_points[0]).y,
    )
    corner_right = (
        facial_landmarks.part(eye_points[3]).x,
        facial_landmarks.part(eye_points[3]).y,
    )

    center_top = midpoint(
        facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2])
    )
    center_bottom = midpoint(
        facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4])
    )

    horizontal_length = euclidean_distance(corner_left, corner_right)
    vertical_length = euclidean_distance(center_top, center_bottom)

    ratio = horizontal_length / vertical_length

    return ratio

# OpenCV setup
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]

blink_counter = 0
start_time = datetime.now()
last_blink_time = start_time

def show_popup_message(message):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showinfo("Blink Alert", message)
    root.destroy()

def update_frame():
    global last_blink_time, blink_counter, start_time  # Declare last_blink_time and blink_counter as global variables
    current_time = datetime.now()  # Initialize current_time here
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        label.config(image=photo)
        label.image = photo

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces, _, _ = detector.run(image=frame_gray, upsample_num_times=0, adjust_threshold=0.0)

        for face in faces:
            landmarks = predictor(frame_gray, face)

            left_eye_ratio = get_blink_ratio(left_eye_landmarks, landmarks)
            right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
            blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

            current_time = datetime.now()

            elapsed_time = (current_time - start_time).total_seconds()

            if elapsed_time >= BLINK_DURATION:
                # Calculate and print the average blink count
                average_blinks = blink_counter / BLINK_DURATION
                print(f"Average Blinks in {BLINK_DURATION} seconds: {average_blinks:.2f}")

                # Reset elapsed time and blink count
                start_time = current_time

                if blink_counter < MIN_BLINK_COUNT:
                    message = f"Low Blink Count ({average_blinks:.2f}) after {BLINK_DURATION} seconds!"
                    show_popup_message(message)
                blink_counter = 0

            if (
                blink_ratio > BLINK_RATIO_THRESHOLD
                and (current_time - last_blink_time).total_seconds() >= COOLDOWN_TIME
            ):
                blink_counter += 1
                last_blink_time = current_time
    # Display blink count and elapsed time in Tkinter labels
    blink_text = f"Blink Count: {blink_counter}"
    elapsed_time = (current_time - start_time).total_seconds()
    elapsed_time_text = f"Elapsed Time: {elapsed_time:.2f}s"
    blink_label.config(text=blink_text)
    elapsed_time_label.config(text=elapsed_time_text)

    if not paused:
        label.after(10, update_frame)  # Refresh every 10 milliseconds

def start_capture():
    global cap, paused, blink_counter, start_time, last_blink_time
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    paused = False
    blink_counter = 0
    start_time = datetime.now()
    last_blink_time = start_time
    update_frame()

def stop_capture():
    global cap, paused
    cap.release()
    paused = True
    label.config(image=None)

root = tk.Tk()
root.title("Blink Detector")

window_width = 640
window_height = 480
root.geometry(f"{window_width}x{window_height}")

label_width = 520
label_height = 400
label = tk.Label(root, width=label_width, height=label_height)
label.pack()

blink_label = tk.Label(root, text="Blink Count: 0")
blink_label.pack()

elapsed_time_label = tk.Label(root, text="Elapsed Time: 0.00s")
elapsed_time_label.pack()

button_width = 20
button_height = 4

button_font_style = font.Font(size=14)

start_button = tk.Button(root, text="Start", command=start_capture, width=button_width, height=button_height, font=button_font_style, fg="green")
stop_button = tk.Button(root, text="Stop", command=stop_capture, width=button_width, height=button_height, font=button_font_style, fg="red")

start_button.pack(side=tk.LEFT)
stop_button.pack(side=tk.RIGHT)

paused = False

update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()

import cv2
import dlib
import math
import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk
import ttkthemes
from PIL import Image, ImageTk

BLINK_RATIO_THRESHOLD = 4
BLINK_DURATION = 60  # Duration in seconds
COOLDOWN_TIME = 1.5  # Cooldown time in seconds
MIN_BLINK_COUNT = 15

blink_history = []  # List to store history data

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

def update_history_table():
    current_time = datetime.now()
    history_item = {
        "Start Time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "End Time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Duration": f"{(current_time - start_time).total_seconds():.2f}s",
        "Blink Count": blink_counter,
    }
    blink_history.append(history_item)
    history_table.insert("", "end", values=(history_item["Start Time"], history_item["End Time"], history_item["Duration"], history_item["Blink Count"]))
    history_table.see(history_table.get_children()[-1])  # Scroll to the last item

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
    update_history_table()

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
    if start_time is not None:
        update_history_table()
    cap.release()
    paused = True
    label.config(image=None)

root = tk.Tk()
root.title("Blink Detector")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Main")
             
style = ttkthemes.ThemedStyle(root)
style.theme_use("arc")

window_width = 640
window_height = 580
root.geometry(f"{window_width}x{window_height}")

blink_label = ttk.Label(root, text="Blink Count: 0")
blink_label.pack()

elapsed_time_label = ttk.Label(root, text="Elapsed Time: 0.00s")
elapsed_time_label.pack()

button_width = 20
button_height = 4

start_button = ttk.Button(root, text="Start", command=start_capture, width=button_width, style="TButton")
stop_button = ttk.Button(root, text="Stop", command=stop_capture, width=button_width, style="TButton")

start_button.pack(side=tk.LEFT)
stop_button.pack(side=tk.RIGHT)

label_width = 480
label = ttk.Label(tab1, text="Main", width=label_width)
label.pack()

tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="History")

history_table = ttk.Treeview(tab2, columns=("Start Time", "End Time", "Duration", "Blink Count"), show="headings")

history_table.column("Start Time", anchor="center", width=100)
history_table.column("End Time", anchor="center", width=100)
history_table.column("Duration", anchor="center", width=80)
history_table.column("Blink Count", anchor="center", width=80)

history_table.heading("Start Time", text="Start Time", anchor="center")
history_table.heading("End Time", text="End Time", anchor="center")
history_table.heading("Duration", text="Duration", anchor="center")
history_table.heading("Blink Count", text="Blink Count", anchor="center")

history_table.pack(fill="both", expand=True)

paused = False

update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()

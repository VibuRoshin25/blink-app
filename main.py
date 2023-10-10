import cv2
import dlib
import math
from datetime import datetime

# Constants
BLINK_RATIO_THRESHOLD = 4
BLINK_DURATION = 60  # Duration in seconds
COOLDOWN_TIME = 0.2  # Cooldown time in seconds


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
cv2.namedWindow("BlinkDetector")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]

blink_counter = 0
start_time = datetime.now()
last_blink_time = start_time

while True:
    retval, frame = cap.read()

    if not retval:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces, _, _ = detector.run(image=frame, upsample_num_times=0, adjust_threshold=0.0)

    for face in faces:
        landmarks = predictor(frame, face)

        left_eye_ratio = get_blink_ratio(left_eye_landmarks, landmarks)
        right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
        blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

        current_time = datetime.now()

        if (
            blink_ratio > BLINK_RATIO_THRESHOLD
            and (current_time - last_blink_time).total_seconds() >= COOLDOWN_TIME
        ):
            blink_counter += 1
            last_blink_time = current_time

        # Display blink count and elapsed time
        blink_text = f"Blink Count: {blink_counter}"
        elapsed_time = (current_time - start_time).total_seconds()
        elapsed_time_text = f"Elapsed Time: {elapsed_time:.2f}s"
        cv2.putText(
            frame,
            blink_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            elapsed_time_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    cv2.imshow("BlinkDetector", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

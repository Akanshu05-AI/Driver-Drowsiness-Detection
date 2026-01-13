import cv2
import dlib
from scipy.spatial import distance as dist
import pygame

# --- Sound Playing Library Choice ---
# Option 2: Using pygame (Cross-platform, requires installation: pip install pygame)
pygame.mixer.init()
sound_channel = None  # Initialize a sound channel
sound = None
try:
    sound = pygame.mixer.Sound("alarm.wav")
except pygame.error as e:
    print(f"Error loading sound with pygame: {e}")

def play_sound():
    global sound_channel, sound
    if sound and (sound_channel is None or not sound_channel.get_busy()):
        sound_channel = sound.play(-1)  # Loop indefinitely

def stop_sound():
    global sound_channel
    if sound_channel and sound_channel.get_busy():
        sound_channel.stop()

# --- End of Sound Playing Library Choice ---

# Load the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the indices for the left and right eye landmarks
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))

# Define a threshold for EAR to indicate drowsiness
EYE_AR_THRESHOLD = 0.25
# Define the number of consecutive frames the eye must be below the threshold
EYE_AR_CONSEC_FRAMES = 48
# Counter for consecutive frames below the threshold
COUNTER = 0
# Boolean flag to indicate if the alarm is on
ALARM_ON = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    drowsiness_level = "Awake"
    status_color = (0, 255, 0)  # Green

    for face in faces:
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        leftEye = [shape[i] for i in LEFT_EYE_POINTS]
        rightEye = [shape[i] for i in RIGHT_EYE_POINTS]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        avgEAR = (leftEAR + rightEAR) / 2.0

        # Draw the eye landmarks
        for (x, y) in leftEye:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
        for (x, y) in rightEye:
            cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 255), -1)

        # Check for drowsiness
        if avgEAR < EYE_AR_THRESHOLD:
            COUNTER += 1
            progress = int((COUNTER / EYE_AR_CONSEC_FRAMES) * 100) # Calculate progress percentage
            if COUNTER > EYE_AR_CONSEC_FRAMES // 3:
                drowsiness_level = "Slightly Drowsy"
                status_color = (0, 255, 255)  # Yellow
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                drowsiness_level = "Very Drowsy"
                status_color = (0, 0, 255)  # Red
                if not ALARM_ON:
                    ALARM_ON = True
                    print("DROWSINESS ALERT!")
                    play_sound()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            COUNTER = 0
            progress = 0
            if ALARM_ON:
                ALARM_ON = False
                stop_sound()

        # --- Visual Indicators ---
        cv2.putText(frame, "Status: {}".format(drowsiness_level), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

        cv2.putText(frame, "EAR: {:.2f}".format(avgEAR), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # --- Progress Bar ---
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = 120
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2) # Outline
        fill_width = int((progress / 100) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), status_color, -1) # Fill

        cv2.putText(frame, "Progress: {}%".format(progress), (bar_x, bar_y + bar_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
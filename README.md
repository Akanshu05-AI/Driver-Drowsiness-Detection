# Real-Time Drowsiness Detection System
I built this project to tackle the serious issue of driver fatigue using computer vision. The system monitors a driver's eyes through a webcam and triggers an alarm if it detects that the user is falling asleep.

## How I built it
The core of this project is based on the **Eye Aspect Ratio (EAR)**. Using **Dlib's 68-point facial landmark predictor**, the script locates the eyes and calculates the ratio between their vertical and horizontal distances.
* **Logic:** When your eyes close, the EAR value drops significantly.
* **Thresholds:** I set the system to trigger a "Very Drowsy" alert if the EAR stays below **0.25** for **48 consecutive frames** (about 2 seconds of video).
* **Alerts:** I used **Pygame** to handle the audio because it allows for a smooth, looping alarm sound.

## Features I included
* **Live Feedback:** I added a real-time EAR tracker on the screen so you can see exactly how the system sees your eyes.
* **Progress Bar:** Instead of just a sudden alarm, I built a visual progress bar that fills up as the system detects drowsiness.
* **Three Levels of Safety:** The status changes from **Awake** (Green) to **Slightly Drowsy** (Yellow) before hitting the **Very Drowsy** (Red) alarm state.

## Tools & Libraries
* **OpenCV:** For the video feed and image processing.
* **Dlib:** To get the facial landmarks.
* **Pygame:** For the alarm sound management.
* **SciPy:** To calculate the Euclidean distance between eye points.

## Getting it running
1. Make sure you have the landmark file `shape_predictor_68_face_landmarks.dat` and an `alarm.wav` in the main folder.
2. Install the requirements:
`pip install opencv-python dlib scipy pygame`
3. Run the script:
`python level_drowsiness.py`


**Would you like me to help you write a short "About" blurb for the sidebar that sounds like a personal summary?**

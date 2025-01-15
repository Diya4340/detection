import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import winsound  # For playing sound (Windows only)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Eye landmarks (referencing MediaPipe Face Mesh indices)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Utility function to calculate eye openness
def calculate_eye_openness(eye_landmarks, landmarks):
    top = landmarks[eye_landmarks[1]]
    bottom = landmarks[eye_landmarks[5]]
    left = landmarks[eye_landmarks[0]]
    right = landmarks[eye_landmarks[3]]

    vertical_distance = np.linalg.norm(np.array(top) - np.array(bottom))
    horizontal_distance = np.linalg.norm(np.array(left) - np.array(right))
    return vertical_distance / horizontal_distance

# Attention logging function
def log_attention(student_id, status):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{timestamp} | {student_id}: {status}")

# Start capturing video
cap = cv2.VideoCapture(0)

# Variables for attention monitoring
attention_threshold = 0.21  # Threshold for determining attentiveness
alert_duration_threshold = 3  # Duration (in seconds) before triggering alert
inattentive_start_times = {}  # Dictionary to track inattentive start times per student

# Variables for student identification
student_id_counter = 1
student_ids = {}  # Map to assign unique IDs to detected faces

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (required for MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Face Mesh
    results = face_mesh.process(rgb_frame)
    current_time = time.time()

    if results.multi_face_landmarks:
        for face_index, face_landmarks in enumerate(results.multi_face_landmarks):
            # Assign a unique ID to each detected face
            if face_index not in student_ids:
                student_ids[face_index] = f"Student-{student_id_counter}"
                student_id_counter += 1

            student_id = student_ids[face_index]

            # Extract landmarks for the current face
            landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark]

            # Calculate eye openness for both eyes
            left_eye_openness = calculate_eye_openness(LEFT_EYE, landmarks)
            right_eye_openness = calculate_eye_openness(RIGHT_EYE, landmarks)

            # Determine attentiveness
            if left_eye_openness > attention_threshold and right_eye_openness > attention_threshold:
                attention_status = "Attentive"
                color = (0, 255, 0)  # Green for attentive
                inattentive_start_times[student_id] = None  # Reset inattentiveness timer
            else:
                attention_status = "Not Attentive"
                color = (0, 0, 255)  # Red for not attentive

                # Start or update inattentiveness timer
                if student_id not in inattentive_start_times or inattentive_start_times[student_id] is None:
                    inattentive_start_times[student_id] = current_time
                elif current_time - inattentive_start_times[student_id] > alert_duration_threshold:
                    # Trigger an alert
                    winsound.Beep(1000, 500)  # Play a beep sound
                    cv2.putText(frame, "ALERT: Pay Attention!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Log the attention status
            log_attention(student_id, attention_status)

            # Display student ID and attention status
            x, y = int(landmarks[1][0]), int(landmarks[1][1])
            cv2.putText(frame, f"{student_id}: {attention_status}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw eye landmarks
            for eye in [LEFT_EYE, RIGHT_EYE]:
                for idx in eye:
                    ex, ey = int(landmarks[idx][0]), int(landmarks[idx][1])
                    cv2.circle(frame, (ex, ey), 2, color, -1)
    else:
        # Clear student IDs and timers if no faces are detected
        student_ids.clear()
        inattentive_start_times.clear()

    # Display the video feed
    cv2.imshow("Virtual Classroom Attention Monitor", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to count raised fingers
def count_fingers(hand_landmarks):
    fingers = 0
    # Thumb: Check if tip is on the right side of the hand for right hand
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers += 1  # Thumb is up

    # For fingers 2 to 5
    # Check each finger by comparing the position of the tip and the dip
    for fingertip, dip in zip(
        [
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP,
        ],
        [
            mp_hands.HandLandmark.INDEX_FINGER_DIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
            mp_hands.HandLandmark.RING_FINGER_DIP,
            mp_hands.HandLandmark.PINKY_DIP,
        ],
    ):
        # Count finger if the tip is above the dip
        if hand_landmarks.landmark[fingertip].y < hand_landmarks.landmark[dip].y:
            fingers += 1  # Finger is up

    return fingers  # Return total count of raised fingers


# Initialize Mediapipe Hands solution
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_fingers_in_live_video():
    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Convert the frame to RGB (required by Mediapipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame with Mediapipe Hands
        result = hands.process(rgb_frame)

        # Draw hand landmarks and count fingers
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers_count = count_fingers(hand_landmarks)
                # Display the count on the frame
                cv2.putText(frame, f'Fingers: {fingers_count}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Finger Count', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

# Run the live finger detection
detect_fingers_in_live_video()

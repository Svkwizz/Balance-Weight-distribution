import mediapipe as mp
import cv2
import numpy as np
import pandas as pd

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create empty lists to store data
com_data = []
weight_distribution_data = []
balance_status_data = []

# Open video file
cap = cv2.VideoCapture("E:\Research\Clinical video\Topcorner.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect poses in the frame
    results = pose.process(frame_rgb)

    # If poses are detected
    if results.pose_landmarks:
        # Extract landmark points
        landmarks = results.pose_landmarks.landmark

        # Calculate center of mass (COM)
        com_x = sum(l.x for l in landmarks) / len(landmarks)
        com_y = sum(l.y for l in landmarks) / len(landmarks)
        com_data.append((com_x, com_y))

        # Calculate weight distribution
        left_foot = np.array(
            [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y])
        right_foot = np.array(
            [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y])
        com = np.array([com_x, com_y])

        # Calculate distance between COM and left and right foot
        distance_to_left_foot = np.linalg.norm(com - left_foot)
        distance_to_right_foot = np.linalg.norm(com - right_foot)

        # Calculate weight distribution ratio
        total_distance = distance_to_left_foot + distance_to_right_foot
        weight_distribution = {
            'left_foot': distance_to_left_foot / total_distance,
            'right_foot': distance_to_right_foot / total_distance
        }
        weight_distribution_data.append(weight_distribution)

        # Assess balance and stability
        # Example: If weight distribution is balanced, COM should be close to the midpoint between the feet
        balance_threshold = 0.1
        if abs(weight_distribution['left_foot'] - weight_distribution['right_foot']) < balance_threshold:
            balance_status = "Balanced"
        else:
            balance_status = "Unbalanced"
        balance_status_data.append(balance_status)

    # Display frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Create a DataFrame from collected data
df = pd.DataFrame({'Center_of_Mass': com_data,
                   'Weight_Distribution': weight_distribution_data,
                   'Balance_Status': balance_status_data})

# Export DataFrame to Excel
df.to_excel('shot_put_data.xlsx', index=False)

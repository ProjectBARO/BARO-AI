import cv2
import numpy as np
import mediapipe as mp
from collections import Counter



mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def calculate_vertical_distance_cm(landmark1, landmark2, frame_height, distance_to_camera_cm=60, camera_fov_degrees=25):
    if landmark1 is None or landmark2 is None:
        return None
    landmark1_pixel = landmark1[1] * frame_height
    landmark2_pixel = landmark2[1] * frame_height
    pixel_distance = np.abs(landmark1_pixel - landmark2_pixel)
    real_height_cm = 2 * distance_to_camera_cm * np.tan(np.radians(camera_fov_degrees / 2))
    cm_per_pixel = real_height_cm / frame_height
    return pixel_distance * cm_per_pixel

def calculate_angle(p1, p2):
    angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    if angle < 0:
        angle += 360
    return angle

def adjust_angle(angle):
    if angle > 180:
        angle = 360 - angle
    return angle

def evaluate_angle_condition(angle):
    adjusted_angle = adjust_angle(angle)
    if 165 <= adjusted_angle <= 180:
        return 'Fine'
    elif 150 <= adjusted_angle < 165:
        return 'Danger'
    elif 135 <= adjusted_angle < 150:
        return 'Serious'
    elif adjusted_angle < 135:
        return 'Very Serious'

def extract_frames(video_file: str, interval: int = 5):
    cap = cv2.VideoCapture(video_file)
    frameRate = cap.get(5)
    images = []
    landmarks_info = []
    angle_conditions = []

    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % (frameRate * interval) == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_frame, (28, 28))
            normalized_img = resized_img / 255.0
            normalized_img = np.stack((normalized_img,) * 3, axis=-1)
            images.append(normalized_img)

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                left_ear = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y]

                vertical_distance_cm = calculate_vertical_distance_cm(left_shoulder, left_ear, frame.shape[0])
                angle = calculate_angle(left_ear, left_shoulder)
                adjusted_angle = adjust_angle(angle)
                angle_status = evaluate_angle_condition(adjusted_angle)

                landmarks_info.append((left_shoulder, left_ear, vertical_distance_cm, adjusted_angle))
                angle_conditions.append(angle_status)

    status_frequencies = Counter(angle_conditions)
    cap.release()
    return np.array(images), landmarks_info, dict(status_frequencies)
import os
import requests
import tempfile
import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, render_template, request, Response
import json
import mediapipe as mp
import math


tf.config.run_functions_eagerly(True)

app = Flask(__name__)

model = tf.keras.models.load_model('CNN_model.h5')
model.compile(run_eagerly=True)

def calculate_acromion_distance_cm(left_shoulder, left_ear, frame_width, frame_height, distance_to_camera_cm=50, camera_fov_degrees=60):
    if left_shoulder is None or left_ear is None:
        return None
    # 픽셀 단위로 변환
    left_shoulder_pixel = [left_shoulder[0] * frame_width, left_shoulder[1] * frame_height]
    left_ear_pixel = [left_ear[0] * frame_width, left_ear[1] * frame_height]
    pixel_distance = np.sqrt((left_shoulder_pixel[0] - left_ear_pixel[0])**2 + (left_shoulder_pixel[1] - left_ear_pixel[1])**2)
    real_width_cm = 2 * distance_to_camera_cm * np.tan(np.radians(camera_fov_degrees / 2))
    cm_per_pixel = real_width_cm / frame_width
    distance_cm = pixel_distance * cm_per_pixel
    return distance_cm

def extract_frames(video_file, interval=5):
    cap = cv2.VideoCapture(video_file)
    frameRate = cap.get(5)
    images = []
    landmarks_info = []

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % (frameRate * interval) == 0:
            img = cv2.resize(frame, (28, 28))
            img = img / 255.0
            images.append(img)

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                left_ear = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y]
                acromion_distance_cm = calculate_acromion_distance_cm(left_shoulder, left_ear, frame.shape[1], frame.shape[0])
                landmarks_info.append((left_shoulder, left_ear, acromion_distance_cm))
            else:
                landmarks_info.append((None, None, None))

    cap.release()
    return np.array(images), landmarks_info

def download_video(url):
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name
    return None

def calculate_posture_ratios(predictions):
    hunched_posture_label = 0
    normal_posture_label = 1

    total_predictions = len(predictions)
    hunched_count = np.sum(predictions == hunched_posture_label)
    normal_count = np.sum(predictions == normal_posture_label)

    hunched_ratio = (hunched_count / total_predictions) * 100
    normal_ratio = (normal_count / total_predictions) * 100

    return hunched_ratio, normal_ratio

def calculate_scores(predictions_proba):
    scores = np.max(predictions_proba, axis=1) * 100  
    return scores.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        video_url = request.args.get('video_url') if request.method == 'GET' else request.form.get('video_url')
        if not video_url:
            raise ValueError("No video URL provided")

        video_file = download_video(video_url)
        if video_file is None:
            raise ValueError("Video download failed")

        images, landmarks_info = extract_frames(video_file)
        predictions_proba = model.predict(images)
        result = np.argmax(predictions_proba, axis=1)
        scores = calculate_scores(predictions_proba)

        hunched_ratio, normal_ratio = calculate_posture_ratios(result)

        os.remove(video_file)

        response_data = json.dumps({
            'result': result.tolist(),
            'hunched_ratio': hunched_ratio,
            'normal_ratio': normal_ratio,
            'scores': scores,
            'landmarks_info': landmarks_info
        }, indent=4)

        return Response(response_data, mimetype='application/json')

    except Exception as e:
        error_message = json.dumps({'error': str(e)}, indent=4)
        return Response(error_message, status=400, mimetype='application/json')

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
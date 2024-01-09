import os
import requests
import tempfile
import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify


# TensorFlow Eager Execution 활성화
tf.config.run_functions_eagerly(True)

app = Flask(__name__)


# 모델 불러오기 및 컴파일
model = tf.keras.models.load_model('CNN_model.h5')
model.compile(run_eagerly=True)


def extract_frames(video_file, interval=5):
    cap = cv2.VideoCapture(video_file)
    frameRate = cap.get(5)  # 프레임 5초 단위
    images = []

    while cap.isOpened():
        frameId = cap.get(1)  # 현재 프레임 번호
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % (frameRate * interval) == 0:
            img = cv2.resize(frame, (28, 28))
            img = img / 255.0
            images.append(img)

    cap.release()
    return np.array(images)

def download_video(url):
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name
    return None


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

        images = extract_frames(video_file)
        predictions = model.predict(images)
        result = np.argmax(predictions, axis=1)

        os.remove(video_file)

        return jsonify({'result': result.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400  
    

    
if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
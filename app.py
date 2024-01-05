from flask import Flask, render_template
import cv2
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# 모델 불러오기
model = tf.keras.models.load_model('CNN_model.h5')

def extract_frames(video_file, interval=5):
    cap = cv2.VideoCapture(video_file)
    frameRate = cap.get(5) # 프레임 5초 단위
    images = []

    while cap.isOpened():
        frameId = cap.get(1) # 현재 프레임 번호
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % (frameRate * interval) == 0:
            img = cv2.resize(frame, (28, 28))
            img = img / 255.0
            images.append(img)

    cap.release()
    return np.array(images)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    video_path = 'pose_estimation_wrong.mp4' 

    try:
        images = extract_frames(video_path)
        predictions = model.predict(images)
        result = np.argmax(predictions, axis=1)

        return render_template('index.html', result=result.tolist())

    except Exception as e:
        return render_template('index.html', result=str(e))

if __name__ == '__main__':
    app.run('0.0.0.0', port = 5000, debug=True)
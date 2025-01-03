import tensorflow as tf
import numpy as np



model = tf.keras.models.load_model('final_baro_model.h5')

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

def predict_posture(images):
    predictions_proba = model.predict(images)
    result = np.argmax(predictions_proba, axis=1)
    scores = calculate_scores(predictions_proba)
    hunched_ratio, normal_ratio = calculate_posture_ratios(result)
    return result, scores, hunched_ratio, normal_ratio
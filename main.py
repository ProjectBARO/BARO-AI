from fastapi import FastAPI, HTTPException, Request
from video_downloader import download_video
from frame_extractor import extract_frames
from model_predictor import predict_posture
from result_analyzer import analyze_results
import os



app = FastAPI()

@app.post("/predict")
async def predict(video_data: dict):
    try:
        video_url = video_data.get("video_url")
        if not video_url:
            raise HTTPException(status_code=400, detail="Missing video_url in request body")
        
        video_file = await download_video(video_url)
        if video_file is None:
            raise HTTPException(status_code=400, detail="Video download failed")
        
        images, landmarks_info, status_frequencies = extract_frames(video_file)
        result, scores, hunched_ratio, normal_ratio = predict_posture(images)

        os.remove(video_file)

        return analyze_results(result, scores, hunched_ratio, normal_ratio, landmarks_info, status_frequencies)
    except Exception as e:
        return {"error": str(e)}
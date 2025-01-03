from fastapi.responses import JSONResponse



def analyze_results(result, scores, hunched_ratio, normal_ratio, landmarks_info, status_frequencies):
    return JSONResponse(content={
        'result': result.tolist(),
        'hunched_ratio': hunched_ratio,
        'normal_ratio': normal_ratio,
        'scores': scores,
        'landmarks_info': [
            {
                'left_shoulder': {'x': info[0][0], 'y': info[0][1]},
                'left_ear': {'x': info[1][0], 'y': info[1][1]},
                'vertical_distance_cm': info[2],
                'angle': info[3]
            } for info in landmarks_info
        ],
        'status_frequencies': status_frequencies
    })
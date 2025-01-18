from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

app = Flask(__name__)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

cap = cv2.VideoCapture(0)

def detect_emotion(face_crop):
    try:
        analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except Exception as e:
        print(f"Error: {e}")
        return "Unknown"

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame!")
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_detection = face_detection.process(frame_rgb)
            people_count = 0
            if results_detection.detections:
                for detection in results_detection.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    face_crop = frame[y:y + h, x:x + w]
                    if face_crop.size != 0:
                        emotion = detect_emotion(face_crop)
                        cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        people_count += 1

            cv2.putText(frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

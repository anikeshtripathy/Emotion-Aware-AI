from fastapi import FastAPI, File, UploadFile, Form
from deepface import DeepFace
from mtcnn import MTCNN
from transformers import pipeline
from pydub import AudioSegment
import speech_recognition as sr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2
from collections import Counter
import time
import os
import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context

app = FastAPI()

# Facial Emotion API
@app.post("/analyze-face/")
async def analyze_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    detector = MTCNN()
    faces = detector.detect_faces(frame)
    if not faces:
        return {"error": "No face detected"}
    
    x, y, w, h = faces[0]["box"]
    cropped_face = frame[y:y + h, x:x + w]
    analysis = DeepFace.analyze(cropped_face, actions=['emotion'], enforce_detection=False)
    dominant_emotion = analysis[0]["dominant_emotion"]
    return {"emotion": dominant_emotion}

# Textual Emotion API
@app.post("/analyze-text/")
async def analyze_text(text: str = Form(...)):
    emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    results = emotion_analyzer(text)
    highest_emotion = max(results[0], key=lambda x: x['score'])
    return {"emotion": highest_emotion['label']}

# Vocal Emotion API
@app.post("/analyze-audio/")
async def analyze_audio(file: UploadFile = File(...)):
    audio_path = f"temp_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())
    
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(audio_path).set_channels(1).set_frame_rate(16000)
    audio.export("temp_audio.wav", format="wav")
    
    with sr.AudioFile("temp_audio.wav") as source:
        audio_data = recognizer.record(source)
    
    try:
        transcribed_text = recognizer.recognize_google(audio_data)
        os.remove("temp_audio.wav")
        os.remove(audio_path)
        return await analyze_text(text=transcribed_text)
    except:
        return {"emotion": "neutral"}

# Unified Emotion Analysis API
@app.post("/analyze-all/")
async def analyze_all(face_file: UploadFile = File(...), audio_file: UploadFile = File(...), text: str = Form(...)):
    facial_emotion = await analyze_face(face_file)
    vocal_emotion = await analyze_audio(audio_file)
    textual_emotion = await analyze_text(text)
    
    emotions = ['happy', 'sad', 'angry', 'neutral', 'fear', 'surprise', 'disgust']
    le = LabelEncoder().fit(emotions)
    X_train = [
        [le.transform(['happy'])[0], le.transform(['neutral'])[0], le.transform(['sad'])[0]],
        [le.transform(['sad'])[0], le.transform(['angry'])[0], le.transform(['neutral'])[0]],
        [le.transform(['neutral'])[0], le.transform(['happy'])[0], le.transform(['surprise'])[0]],
    ]
    y_train = le.transform(['happy', 'sad', 'neutral'])
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    input_features = np.array([
        le.transform([facial_emotion['emotion']])[0] if facial_emotion['emotion'] in emotions else le.transform(['neutral'])[0],
        le.transform([vocal_emotion['emotion']])[0] if vocal_emotion['emotion'] in emotions else le.transform(['neutral'])[0],
        le.transform([textual_emotion['emotion']])[0] if textual_emotion['emotion'] in emotions else le.transform(['neutral'])[0],
    ]).reshape(1, -1)
    
    predicted_label = clf.predict(input_features)
    final_emotion = le.inverse_transform(predicted_label)[0]
    
    return {
        "facial_emotion": facial_emotion["emotion"],
        "vocal_emotion": vocal_emotion["emotion"],
        "textual_emotion": textual_emotion["emotion"],
        "final_emotion": final_emotion
    }

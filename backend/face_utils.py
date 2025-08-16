import os
import cv2
import numpy as np
from typing import List, Tuple

BASE_BACKEND = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_BACKEND, '..', 'data'))
FACES_DIR = os.path.join(DATA_DIR, 'faces')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'lbph.yml')

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

# Create recognizer
RECOGNIZER = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)


def _detect_face(gray):
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    return faces


def capture_samples(student_id: int, samples: int = 20) -> str:
    """Capture N face samples for a student. Returns a preview image path."""
    folder = os.path.join(FACES_DIR, str(student_id))
    os.makedirs(folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    preview_path = None

    while count < samples:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detect_face(gray)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))
            filepath = os.path.join(folder, f"{student_id}_{count:03d}.png")
            cv2.imwrite(filepath, roi)
            preview_path = filepath
            count += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"Captured {count}/{samples}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Capture Samples - Press 'q' to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return preview_path or ""


def _load_dataset() -> Tuple[List[np.ndarray], List[int]]:
    images, labels = [], []
    for sid in os.listdir(FACES_DIR):
        sid_path = os.path.join(FACES_DIR, sid)
        if not os.path.isdir(sid_path):
            continue
        for f in os.listdir(sid_path):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img = cv2.imread(os.path.join(sid_path, f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(img)
                labels.append(int(sid))
    return images, labels


def train_or_update_model() -> bool:
    images, labels = _load_dataset()
    if not images:
        return False
    RECOGNIZER.train(images, np.array(labels))
    RECOGNIZER.save(MODEL_PATH)
    return True


def ensure_model_loaded() -> bool:
    if not os.path.exists(MODEL_PATH):
        return False
    RECOGNIZER.read(MODEL_PATH)
    return True


def recognize_once(threshold: float = 70.0) -> int | None:
    """Open webcam, detect a face, predict student_id. Returns None if no confident match."""
    if not ensure_model_loaded():
        # Try to train from any available data
        if not train_or_update_model():
            return None

    cap = cv2.VideoCapture(0)
    predicted = None
    best_conf = 999.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detect_face(gray)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))
            label, confidence = RECOGNIZER.predict(roi)
            # Lower confidence == better match for LBPH
            text = f"ID:{label} conf:{confidence:.1f}"
            color = (0,255,0) if confidence < threshold else (0,0,255)
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if confidence < best_conf and confidence < threshold:
                best_conf = confidence
                predicted = label
        cv2.imshow("Mark Attendance - Press 'q' to confirm", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return predicted
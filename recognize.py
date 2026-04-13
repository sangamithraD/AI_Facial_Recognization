import pickle
import numpy as np
from deepface import DeepFace

DB_FILE = "employee_embeddings.pkl"

# Load database
with open(DB_FILE, "rb") as f:
    employee_db = pickle.load(f)

# Compute cosine similarity between two embeddings
def cosine_similarity(a, b):  
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Recognize employee from image path
def recognize_face(image_path, threshold=0.65):
    # Generate embedding for the query image
    query_embedding = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet512",
        enforce_detection=False
    )[0]["embedding"]

    best_emp = None
    best_score = -1

    # Compare with all embeddings in the database
    for emp_id, stored_embedding in employee_db.items():
        score = cosine_similarity(query_embedding, stored_embedding)
        if score > best_score:
            best_score = score
            best_emp = emp_id

    if best_score >= threshold:
        return {
            "matched": True,
            "employee_id": best_emp,
            "confidence": round(float(best_score), 2)
        }

    return {
        "matched": False,
        "employee_id": None,
        "confidence": round(float(best_score), 2)
    }
    

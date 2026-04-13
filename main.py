from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
import pickle
from deepface import DeepFace
from recognize import recognize_face

app = FastAPI()

DATASET = "Employee"
DB_FILE = "employee_embeddings.pkl"

# Ensure Employee folder exists
os.makedirs(DATASET, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Attendance Face Recognition API is running"}


# =========================
# Attendance Recognition API
# =========================
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"

    try:
        # Save uploaded selfie temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run face recognition
        result = recognize_face(temp_path)

        return result

    finally:
        # Delete temp image automatically
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =========================
# Add New Employee API
# =========================
@app.post("/add-employee")
async def add_employee(
    employee_id: str = Form(...),
    file: UploadFile = File(...)
):
    file_path = os.path.join(DATASET, f"{employee_id}.jpg")

    # Save employee image permanently
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Generate embedding for new employee
    embedding = DeepFace.represent(
        img_path=file_path,
        model_name="Facenet512",
        enforce_detection=False
    )[0]["embedding"]

    # Load existing embedding DB
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            db = pickle.load(f)
    else:
        db = {}

    # Add/update employee embedding
    db[employee_id] = embedding

    # Save updated DB
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)

    return {
        "status": "success",
        "employee_id": employee_id,
        "message": "Employee added successfully"
    }
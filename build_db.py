from deepface import DeepFace
import os
import pickle

DATASET = "Employee"
DB_FILE = "employee_embeddings.pkl"

# Create empty database dictionary
employee_db = {}

for file in os.listdir(DATASET):
    if file.lower().endswith((".jpg", ".png")):
        emp_id = os.path.splitext(file)[0]
        img_path = os.path.join(DATASET, file)

        # Generate FaceNet embedding
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name="Facenet512",
            enforce_detection=False
        )[0]["embedding"]

        employee_db[emp_id] = embedding

# Save embeddings
with open(DB_FILE, "wb") as f:
    pickle.dump(employee_db, f)

print(f"Embeddings database created with {len(employee_db)} employees.")
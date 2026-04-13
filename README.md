# Attendance Face Recognition API

This is a face recognition-based attendance system built as a FastAPI microservice for office environments. It handles employee onboarding and attendance marking using facial recognition.

The system is designed to integrate with a frontend application and a geotag/location validation service.

---

# Features

- Add new employees using image upload
- Real-time employee recognition using selfie capture
- Face recognition using DeepFace (FaceNet embeddings)
- REST API integration for frontend/backend systems
- Automatic embedding generation and update
- Temporary image cleanup for security

---

# Tech Stack

- Python
- FastAPI
- DeepFace (FaceNet512)
- OpenCV
- NumPy
- Uvicorn

---

# Project Structure

attendance-face-api/

- Employee/                  → Employee dataset images
- build_db.py              → Creates initial embeddings database
- recognize.py             → Face recognition logic
- main.py                  → FastAPI server (API layer)
- employee_embeddings.pkl  → Saved face embeddings database
- requirements.txt         → Python dependencies

---

# Setup Instructions

## 1. Clone repository

git clone <repo-url>
cd attendance-face-api

---

## 2. Install dependencies

pip install -r requirements.txt

---

## 3. Build embeddings database

python build_db.py

This will generate:
employee_embeddings.pkl

---

## 4. Run API server

python -m uvicorn main:app --reload

Server will run at:
http://127.0.0.1:8000

---

## 5. Open API documentation

http://127.0.0.1:8000/docs

---

# API Endpoints

---

# 1. Add New Employee

Used by admin to onboard new employees.

Endpoint:
POST /add-employee

Request Type:
multipart/form-data

Parameters:

- employee_id → string (Employee ID)
- file → image (Employee face image)

Example:
employee_id = EMP001
file = employee.jpg

Response:

{
  "status": "success",
  "employee_id": "EMP001",
  "message": "Employee added successfully"
}

---

# 2. Recognize Employee (Attendance)

Used when employee captures selfie for attendance.

Endpoint:
POST /recognize

Request Type:
multipart/form-data

Parameters:

- file → image (Captured selfie)

Response (Match Found):

{
  "matched": true,
  "employee_id": "EMP001",
  "confidence": 0.95
}

Response (No Match):

{
  "matched": false,
  "employee_id": null,
  "confidence": 0.40
}

---

# System Workflow

## Admin Flow

- Admin uploads employee image
- Calls /add-employee API
- Image is stored in dataset
- Embedding is generated and saved
- Employee becomes available instantly

---

## Attendance Flow

- Employee captures selfie
- Calls /recognize API
- System identifies employee
- Backend combines result with geotag validation
- Attendance is marked

---

# Integration Notes

Frontend responsibilities:
- Capture employee image or selfie
- Call API endpoints
- Display response to user

Backend / Geotag responsibilities:
- Validate location constraints
- Apply attendance rules
- Store attendance records

This service only performs face recognition.

---

# Data Handling

- Employee images → stored in Employee/
- Embeddings → stored in employee_embeddings.pkl
- Temporary images → deleted automatically after recognition

---

# Future Improvements

- Firebase / cloud storage integration
- Employee update and delete APIs
- Liveness detection (anti-spoofing)
- Multi-office scalability
- Docker deployment
- Cloud hosting (AWS / Render / Railway)

---

# Developer Note

This service is a standalone AI microservice focused only on facial recognition.

It is designed to be easily integrated into any frontend or backend system.

Core idea:

Add employee once → recognize anytime using face image

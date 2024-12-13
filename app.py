from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the trained model
model = load_model(r"C:\Users\tessa\Downloads\neuralnetworks\first_year_persistence_nn_model.keras")

# Placeholder scaler
scaler = StandardScaler()

# In-memory database for student details
students_db = {}

# Login credentials for Admin and Student
users = {
    "admin": "adminpass",
    "student": "studentpass"
}

# Define input schema for predictions
class StudentData(BaseModel):
    FirstTermGpa: float
    SecondTermGpa: float
    FirstLanguage: int
    Funding: int
    School: int
    FastTrack: int
    Coop: int
    Residency: int
    Gender: int
    PrevEducation: int
    AgeGroup: int
    HighSchoolAverageMark: float
    MathScore: float
    EnglishGrade: float

class AdminStudentData(StudentData):
    student_id: str  # Unique identifier for each student

# Root route
@app.get("/")
def root():
    """Root route to display information about the API."""
    return {
        "message": "Welcome to the Neural Network API",
        "routes": {
            "/predict": "POST - Send student data to get a prediction",
            "/login": "POST - Login API",
            "/admin/add": "POST - Admin adds a student",
            "/admin/remove/{student_id}": "DELETE - Admin removes a student",
            "/admin/update/{student_id}": "PUT - Admin updates student details",
            "/student/predict/{student_id}": "POST - Student fetches persistence prediction",
            "/datetime": "GET - Fetch current date and time",
        }
    }

# Login API
class LoginData(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(data: LoginData):
    """API for Admin/Student login."""
    if data.username in users and users[data.username] == data.password:
        return {"message": "Login successful", "user_type": data.username}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# Admin APIs
@app.post("/admin/add")
def add_student(student: AdminStudentData, username: str, password: str):
    """Admin API to add a student."""
    if username != "admin" or password != users.get("admin"):
        raise HTTPException(status_code=403, detail="Unauthorized")

    if student.student_id in students_db:
        raise HTTPException(status_code=400, detail="Student already exists")

    students_db[student.student_id] = student.dict()
    return {"message": "Student added successfully", "student_id": student.student_id}

@app.delete("/admin/remove/{student_id}")
def remove_student(student_id: str, username: str, password: str):
    """Admin API to remove a student."""
    if username != "admin" or password != users.get("admin"):
        raise HTTPException(status_code=403, detail="Unauthorized")

    if student_id not in students_db:
        raise HTTPException(status_code=404, detail="Student not found")

    del students_db[student_id]
    return {"message": "Student removed successfully", "student_id": student_id}

@app.put("/admin/update/{student_id}")
def update_student(student_id: str, student: StudentData, username: str, password: str):
    """Admin API to update a student's details."""
    if username != "admin" or password != users.get("admin"):
        raise HTTPException(status_code=403, detail="Unauthorized")

    if student_id not in students_db:
        raise HTTPException(status_code=404, detail="Student not found")

    students_db[student_id] = student.dict()
    return {"message": "Student updated successfully", "student_id": student_id}

# Student API
@app.post("/student/predict/{student_id}")
def predict_for_student(student_id: str, username: str, password: str):
    """Student API to fetch prediction."""
    if username != "student" or password != users.get("student"):
        raise HTTPException(status_code=403, detail="Unauthorized")

    if student_id not in students_db:
        raise HTTPException(status_code=404, detail="Student not found")

    student_data = students_db[student_id]
    input_data = pd.DataFrame([student_data])

    # Scale input data
    scaled_data = scaler.fit_transform(input_data.drop(columns=["student_id"]))
    prediction = model.predict(scaled_data)
    result = "Persist" if prediction[0][0] > 0.5 else "Not Persist"

    return {"student_id": student_id, "prediction": result}

# Date and Time API
@app.get("/datetime")
def get_datetime():
    """API to fetch current date and time."""
    return {"date": datetime.now().strftime("%Y-%m-%d"), "time": datetime.now().strftime("%H:%M:%S")}

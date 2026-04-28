Project Overview:

This project presents an AI-based traffic management system designed to improve traffic flow at road intersections using real-time video analysis. The system detects vehicles, calculates traffic density, and dynamically adjusts traffic signal timing. It also includes an ambulance detection feature to provide immediate priority to emergency vehicles.

The goal of this project is to reduce congestion, minimize waiting time, and make traffic control more efficient using computer vision and machine learning techniques.

Key Features:
Real-time vehicle detection using YOLOv8
Traffic density calculation for each lane
Dynamic traffic signal control based on congestion
Ambulance detection using AI + OCR techniques
Priority signal override for emergency vehicles
Web-based dashboard for live monitoring
Multi-lane traffic analysis

Technologies Used:
Python
OpenCV
YOLOv8 (Ultralytics)
Tesseract OCR
Flask (Web Framework)
HTML, CSS, JavaScript

How It Works:
Traffic video is captured from cameras
Video is processed frame-by-frame
YOLO model detects vehicles in each lane
Vehicles are counted to calculate traffic density
Signal timing is adjusted dynamically
Ambulance detection triggers immediate priority
Output is displayed on a web dashboard

Results:
Improved traffic flow compared to fixed-time signals
Reduced average waiting time
Accurate vehicle detection in real-time
Successful ambulance priority handling
🚀 Future Scope
Traffic violation detection using number plate recognition
Smart surveillance for investigation support
Traffic analytics and reporting (daily/weekly insights)
Integration with IoT and smart city infrastructure
Multi-intersection coordination system

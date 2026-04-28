import cv2
from flask import Response, Flask, render_template, request, redirect, url_for, jsonify
import os
import time
from ultralytics import YOLO
import pytesseract

# Model
model = YOLO("yolov8s.pt")
model.to("cuda")

# GLOBAL DATA
lane_data = {i: {"count": 0, "density": 0, "ambulance": False} for i in range(4)}
lane_stats = {i: {"total_passed": 0} for i in range(4)}
current_cycle_count = {i: 0 for i in range(4)}

uploaded_videos = []

current_lane = 0
last_switch_time = time.time()

# Ambulance Detection
def is_ambulance(vehicle_img, hsv_img):
    if vehicle_img.size == 0:
        return False

    mean_color = vehicle_img.mean()
    is_white = mean_color > 170

    mask1 = cv2.inRange(hsv_img, (0,70,50), (10,255,255))
    mask2 = cv2.inRange(hsv_img, (170,70,50), (180,255,255))

    red_ratio = cv2.countNonZero(mask1 + mask2) / (vehicle_img.shape[0] * vehicle_img.shape[1])
    has_red = red_ratio > 0.04

    return is_white and has_red


# OCR
def detect_text(vehicle_img):
    try:
        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        return text.lower()
    except:
        return ""


# Frame Generator yahan
def generate_frames(video_path, lane):
    global lane_data, current_cycle_count, current_lane

    cap = cv2.VideoCapture(video_path)

    target_fps = 15
    frame_time = 1.0 / target_fps
    frame_count = 0

    while True:
        start_time = time.time()

        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1

        # FRAME SKIP
        if frame_count % 2 != 0:
            continue

        frame = cv2.resize(frame, (640, 360))

        vehicle_count = 0
        lane_data[lane]["ambulance"] = False

        try:
            # Fixed Detection Setting
            results = model(frame, device=0, conf=0.3, imgsz=640)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    # UPDATED CLASS FILTER
                    if label not in ['car', 'motorbike', 'bus', 'truck', 'person']:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    vehicle_count += 1

                    vehicle_img = frame[y1:y2, x1:x2]
                    vehicle_hsv = hsv[y1:y2, x1:x2]

                    if vehicle_img.size == 0:
                        continue

                    ambulance_detected = False

                    # Red Light Detection
                    red_mask1 = cv2.inRange(vehicle_hsv, (0, 70, 50), (10, 255, 255))
                    red_mask2 = cv2.inRange(vehicle_hsv, (170, 70, 50), (180, 255, 255))
                    red_pixels = red_mask1 + red_mask2

                    red_ratio = cv2.countNonZero(red_pixels) / (vehicle_img.shape[0] * vehicle_img.shape[1])

                    # Brightness
                    brightness = vehicle_img.mean()

                    # OCR
                    text = detect_text(vehicle_img)

                    # Final condition
                    if (
                        (red_ratio > 0.015 and brightness > 150)   # BOTH needed
                        or "ambulance" in text
                    ):
                        ambulance_detected = True

                    if label in ['car', 'truck', 'bus'] and ambulance_detected:
                        lane_data[lane]["ambulance"] = True
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
                        cv2.putText(frame, "AMBULANCE", (x1,y1-25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    else:
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                    cv2.putText(frame, label, (x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        except Exception as e:
            print("YOLO Error:", e)

        # density calculation
        lane_capacity = 30
        density = vehicle_count / lane_capacity

        lane_data[lane]["count"] = vehicle_count
        lane_data[lane]["density"] = round(density, 2)

        if lane == current_lane:
            current_cycle_count[lane] = max(current_cycle_count[lane], vehicle_count)

        # stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        sleep_time = frame_time - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)


# Flask
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        if request.form['username']=="Pranjal" and request.form['password']=="998877":
            return redirect(url_for('upload'))
    return render_template('login.html')


@app.route('/upload', methods=['GET','POST'])
def upload():
    global uploaded_videos, lane_data, lane_stats, current_cycle_count

    if request.method == 'POST':
        uploaded_videos = []

        lane_data = {i: {"count":0,"density":0,"ambulance":False} for i in range(4)}
        lane_stats = {i: {"total_passed":0} for i in range(4)}
        current_cycle_count = {i: 0 for i in range(4)}

        for i in range(1,5):
            file = request.files.get(f'lane{i}')
            if file and file.filename:
                path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(path)
                uploaded_videos.append(path)

        return redirect(url_for('dashboard'))

    return render_template('upload.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', videos=uploaded_videos)


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')


@app.route('/video_feed/<int:lane>')
def video_feed(lane):
    if lane < len(uploaded_videos):
        return Response(generate_frames(uploaded_videos[lane], lane),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "No video"


@app.route('/lane_data')
def get_lane_data():
    return jsonify(lane_data)


@app.route('/active_lane')
def active_lane():
    global current_lane, last_switch_time

    now = time.time()

    base_time = 12
    max_extra = 13

    while True:
        density = lane_data[current_lane]["density"]
        green_time = base_time + int(density * max_extra)

        elapsed = now - last_switch_time

        if elapsed < green_time:
            break

        lane_stats[current_lane]["total_passed"] = current_cycle_count[current_lane]
        current_cycle_count[current_lane] = 0

        last_switch_time += green_time
        current_lane = (current_lane + 1) % 4

    density = lane_data[current_lane]["density"]
    green_time = base_time + int(density * max_extra)

    elapsed = now - last_switch_time
    remaining = max(0, int(green_time - elapsed))

    signal = "yellow" if remaining <= 2 else "green"

    return jsonify({
        "lane": current_lane,
        "green_time": green_time,
        "start_time": last_switch_time,
        "remaining": remaining,
        "signal": signal
    })


@app.route('/analysis_data')
def analysis_data():

    total_vehicles = sum(lane_data[i]["count"] for i in lane_data)
    avg_density = sum(lane_data[i]["density"] for i in lane_data)/4

    return jsonify({
        "lanes": lane_data,
        "lane_stats": lane_stats,
        "total_vehicles": total_vehicles,
        "avg_density": round(avg_density,2),
        "precision": "--",
        "recall": "--",
        "f1_score": "--"
    })


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000, threads=8)

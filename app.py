from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2, csv, threading, time, numpy as np
from datetime import datetime
from ultralytics import YOLO
import atexit

app = Flask(__name__)
model = YOLO("yolov8n.pt")

csv_filename = "event_log.csv"
lock = threading.Lock()

# Shared webcam feed
shared_feed = cv2.VideoCapture(0)
time.sleep(1)  # allow camera to warm up

# Cameras list (virtual cameras)
cameras = []
camera_id_counter = 1

# Cleanup on exit
def cleanup():
    if shared_feed is not None and shared_feed.isOpened():
        shared_feed.release()
        shared_feed = None
    cv2.destroyAllWindows()
atexit.register(cleanup)

# CSV logging
def write_csv(camera_id, event, count):
    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), camera_id, event, count])

# ---------------- Flask routes ----------------

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/dashboard")
def dashboard():
    return render_template("index.html")

@app.route("/add_camera", methods=["POST"])
def add_camera():
    global camera_id_counter, shared_feed
    data = request.json
    mode = data["mode"]
    max_people = data.get("max_people")
    if mode == "restricted":
        max_people = None

    # Initialize shared webcam if it's None (after stop)
    if shared_feed is None:
        shared_feed = cv2.VideoCapture(0)
        time.sleep(1)  # give camera time to warm up

    camera_obj = {
        "id": camera_id_counter,
        "mode": mode,
        "max_people": max_people,
        "running": False,
        "alert": None,
        "latest_frame": None
    }

    # Start detection thread
    thread = threading.Thread(target=camera_worker, args=(camera_obj,), daemon=True)
    thread.start()

    with lock:
        cameras.append(camera_obj)
        camera_id_counter += 1

    return jsonify({"status":"added", "camera_id": camera_obj["id"]})

@app.route("/start_event", methods=["POST"])
def start_event():
    global shared_feed
    if shared_feed is None:
        shared_feed = cv2.VideoCapture(0)
        time.sleep(1)  # give webcam time to warm up

    with lock:
        for cam in cameras:
            cam["running"] = True

    # Initialize CSV
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Camera ID", "Event", "People Count"])
    return jsonify({"status":"started"})

@app.route("/stop_event", methods=["POST"])
def stop_event():
    global cameras, shared_feed
    with lock:
        for cam in cameras:
            cam["running"] = False
        cameras = []  # clear all cameras

    # Release shared webcam
    if shared_feed is not None and shared_feed.isOpened():
        shared_feed.release()
        shared_feed = None

    return jsonify({"status":"stopped"})

@app.route("/download")
def download():
    return send_file(csv_filename, as_attachment=True)

@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id):
    cam = next((c for c in cameras if c["id"] == camera_id), None)
    if cam:
        return Response(gen_frames(cam), mimetype="multipart/x-mixed-replace; boundary=frame")
    return "Camera not found", 404

@app.route("/alerts")
def alerts():
    with lock:
        return jsonify([{"camera_id": c["id"], "message": c["alert"]} for c in cameras if c["alert"]])

# ---------------- Camera detection worker ----------------
def camera_worker(camera_obj):
    while True:
        success, frame = shared_feed.read()
        if not success or frame is None:
            time.sleep(0.05)
            continue

        # Optionally resize for speed
        small_frame = cv2.resize(frame, (320, 320))

        if camera_obj["running"]:
            results = model(small_frame, stream=True)
            person_count = sum(1 for r in results for box in r.boxes if int(box.cls)==0)

            # Determine alert
            alert = None
            if camera_obj["mode"]=="restricted" and person_count>0:
                alert = f"Person in restricted area! Count: {person_count}"
                write_csv(camera_obj["id"], "Restricted Area Violation", person_count)
            elif camera_obj["mode"]=="crowd" and camera_obj["max_people"] is not None and person_count > int(camera_obj["max_people"]):
                alert = f"Crowd limit exceeded! Count: {person_count}"
                write_csv(camera_obj["id"], "Crowd Exceeded", person_count)
            camera_obj["alert"] = alert

        # Store latest frame (just raw frame, no polygons)
        _, buffer = cv2.imencode(".jpg", frame)
        camera_obj["latest_frame"] = buffer.tobytes()
        time.sleep(0.05)

# ---------------- Frame generator ----------------
def gen_frames(camera_obj):
    while True:
        if camera_obj["latest_frame"] is not None:
            frame = camera_obj["latest_frame"]
        else:
            # Black placeholder
            blank = np.zeros((480,640,3), dtype=np.uint8)
            _, buffer = cv2.imencode(".jpg", blank)
            frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+frame+b"\r\n")

# ---------------- Run ----------------
if __name__=="__main__":
    app.run(debug=True, threaded=True)
from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2, csv, threading, time, numpy as np
from datetime import datetime
from ultralytics import YOLO
import atexit

app = Flask(__name__)
model = YOLO("yolov8n.pt")

csv_filename = "event_log.csv"
lock = threading.Lock()

shared_feed = cv2.VideoCapture(0)
time.sleep(1)

cameras = []
camera_id_counter = 1

def cleanup():
    global shared_feed
    if shared_feed is not None and shared_feed.isOpened():
        shared_feed.release()
        shared_feed = None
    cv2.destroyAllWindows()
atexit.register(cleanup)

def write_csv(camera_id, event, count):
    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), camera_id, event, count])


@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/dashboard")
def dashboard():
    return render_template("index.html")

@app.route("/add_camera", methods=["POST"])
def add_camera():
    global camera_id_counter, shared_feed
    data = request.json or {}
    mode = data.get("mode")
    if mode not in ("crowd", "restricted"):
        return jsonify({"error": "mode must be 'crowd' or 'restricted'"}), 400

    max_people = None
    if mode == "crowd":
        try:
            max_people = int(data.get("max_people"))
            if max_people < 1:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"error": "max_people must be a positive integer for crowd mode"}), 400

    if shared_feed is None:
        shared_feed = cv2.VideoCapture(0)
        time.sleep(1)

    camera_obj = {
        "id": camera_id_counter,
        "mode": mode,
        "max_people": max_people,
        "running": False,
        "active": True,
        "alert": None,
        "person_count": 0,
        "latest_frame": None
    }

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
        time.sleep(1)

    with lock:
        for cam in cameras:
            cam["running"] = True

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
            cam["active"] = False
        cameras = []

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
        return jsonify([{"camera_id": c["id"], "message": c["alert"], "current_people": c["person_count"]} for c in cameras if c["alert"]])



def camera_worker(camera_obj):
    while camera_obj.get("active", True):
        feed = shared_feed
        if feed is None:
            time.sleep(0.05)
            continue

        success, frame = feed.read()
        if not success or frame is None:
            time.sleep(0.05)
            continue

        small_frame = cv2.resize(frame, (320, 320))

        if camera_obj["running"]:
            results = model(small_frame, stream=True)
            person_count = sum(1 for r in results for box in r.boxes if int(box.cls)==0)
            camera_obj["person_count"] = person_count

            prev_alert = camera_obj["alert"]
            alert = None
            event_type = None
            if camera_obj["mode"]=="restricted" and person_count>0:
                alert = f"Person in restricted area! Count: {person_count}"
                event_type = "Restricted Area Violation"
            elif camera_obj["mode"]=="crowd" and camera_obj["max_people"] is not None and person_count > camera_obj["max_people"]:
                alert = f"Crowd limit exceeded! Count: {person_count}"
                event_type = "Crowd Exceeded"

            if alert and not prev_alert:
                write_csv(camera_obj["id"], event_type, person_count)
            camera_obj["alert"] = alert

        _, buffer = cv2.imencode(".jpg", frame)
        camera_obj["latest_frame"] = buffer.tobytes()
        time.sleep(0.05)

def gen_frames(camera_obj):
    while True:
        if camera_obj["latest_frame"] is not None:
            frame = camera_obj["latest_frame"]
        else:
            blank = np.zeros((480,640,3), dtype=np.uint8)
            _, buffer = cv2.imencode(".jpg", blank)
            frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+frame+b"\r\n")


if __name__=="__main__":
    app.run(debug=True, threaded=True, use_reloader=False)
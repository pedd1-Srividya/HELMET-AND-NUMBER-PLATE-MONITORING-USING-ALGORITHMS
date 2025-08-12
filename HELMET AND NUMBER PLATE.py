import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
import os
from tkinter import Tk, filedialog
import csv
import re
import matplotlib.pyplot as plt
import urllib.request

# --- Function Definitions ---

def is_valid_plate(plate_text):
    pattern = r'^[A-Z0-9]{4,10}$'
    return bool(re.match(pattern, plate_text))

def is_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def download_helmet_model(model_path):
    if not os.path.exists(model_path):
        print("Downloading helmet detection model...")
        url = "https://huggingface.co/keremberke/yolov8n-helmet/resolve/main/yolov8n-helmet.pt"
        headers = {
            "Authorization": "Bearer YOUR_ACCESS_TOKEN"  # <<< Replace with your real Hugging Face token
        }
        
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
                out_file.write(response.read())
            print("Download complete.")
        except urllib.error.HTTPError as e:
            print(f"Failed to download model: {e.code} {e.reason}")

# --- Setup ---

# OCR confidence threshold
try:
    OCR_CONFIDENCE_THRESHOLD = float(input("Enter OCR confidence threshold (e.g., 0.5 for 50%): "))
except ValueError:
    OCR_CONFIDENCE_THRESHOLD = 0.5
    print("Invalid input. Defaulting to 0.5")
print(f"Using OCR confidence threshold: {OCR_CONFIDENCE_THRESHOLD}")

# File dialog to choose video
Tk().withdraw()
video_path = filedialog.askopenfilename(
    title="Select Video File",
    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
)

if not video_path:
    print("No file selected. Exiting.")
    exit()
print("Selected video:", video_path)

# Load models
print("Script started")
plate_model = YOLO(r"C:\Users\peddi\Downloads\license_plate_detector.pt")
helmet_model_path = r"C:\Users\peddi\OneDrive\Desktop\NPR\helmet_detector.pt"
download_helmet_model(helmet_model_path)
helmet_model = YOLO(helmet_model_path)

reader = easyocr.Reader(['en'], gpu=True)

# Create output folders
os.makedirs("output", exist_ok=True)
os.makedirs("output/violations", exist_ok=True)

# Prepare violation logging
violation_log = open("output/violation_log.csv", "w", newline='')
violation_writer = csv.writer(violation_log)
violation_writer.writerow(["Frame", "Violation Type", "Details"])

# Open video
cap = cv2.VideoCapture(video_path)
frame_count = 0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("output/output_with_text.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# --- Metrics ---
total_confidence = 0
total_detections = 0
successful_detections = 0
true_positives = 0
false_positives = 0
false_negatives = 0
frame_accuracy_list = []
frame_loss_list = []
all_confidences = []
seen_plates = set()

# --- Processing Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # License Plate Detection
    plate_results = plate_model(frame)[0]

    if len(plate_results.boxes) == 0:
        print(f"Violation at frame {frame_count}: No plate detected")
        try:
            cv2.imwrite(f"output/violations/no_plate_frame{frame_count}.png", frame)
        except PermissionError as e:
            print(f"PermissionError saving no plate frame {frame_count}. Details: {e}")
        violation_writer.writerow([frame_count, "No Plate Detected", "No bounding boxes found"])

    for i, box in enumerate(plate_results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if box.conf[0] < 0.2:
            continue

        roi = frame[y1:y2, x1:x2]

        if is_blurry(roi):
            print(f"Tampering suspected at frame {frame_count}: Plate is blurry")
            try:
                cv2.imwrite(f"output/violations/blurry_plate_frame{frame_count}_plate{i}.png", roi)
            except PermissionError as e:
                print(f"PermissionError saving blurry plate frame {frame_count}. Details: {e}")
            violation_writer.writerow([frame_count, "Blurry Plate", "Plate appears blurry"])

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        text_results = reader.readtext(resized)
        if text_results:
            plate_text = text_results[0][1]
            confidence = text_results[0][2]
            all_confidences.append(confidence)
            print(f"OCR Text: {plate_text}, OCR Confidence: {confidence:.2f}")

            if plate_text not in seen_plates and plate_text != "Low Confidence" and plate_text != "No Text":
                seen_plates.add(plate_text)
                try:
                    cv2.imwrite(f"output/plates_frame{frame_count}_plate{i}.png", roi)
                except PermissionError as e:
                    print(f"PermissionError saving plate frame {frame_count}. Details: {e}")

            if confidence > OCR_CONFIDENCE_THRESHOLD and is_valid_plate(plate_text):
                total_confidence += confidence
                successful_detections += 1
                ground_truth = "ABC123"
                if plate_text == ground_truth:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if confidence < OCR_CONFIDENCE_THRESHOLD:
                    print(f"Low OCR confidence at frame {frame_count}")
                    violation_writer.writerow([frame_count, "Low OCR Confidence", f"OCR Confidence: {confidence:.2f}"])
                    try:
                        cv2.imwrite(f"output/violations/low_confidence_frame{frame_count}_plate{i}.png", roi)
                    except PermissionError as e:
                        print(f"PermissionError saving low confidence plate {frame_count}. Details: {e}")

                if not is_valid_plate(plate_text):
                    print(f"Invalid plate format '{plate_text}' at frame {frame_count}")
                    violation_writer.writerow([frame_count, "Invalid Plate Format", plate_text])
                    try:
                        cv2.imwrite(f"output/violations/invalid_plate_frame{frame_count}_plate{i}.png", roi)
                    except PermissionError as e:
                        print(f"PermissionError saving invalid plate {frame_count}. Details: {e}")

                plate_text=""
                confidence = 40.0
                false_negatives += 1
        else:
            plate_text=""
            confidence = 0.0
            false_negatives += 1

        total_detections += 1
        frame_accuracy_list.append(confidence if confidence > OCR_CONFIDENCE_THRESHOLD else 0.0)
        frame_loss_list.append(1 - confidence if confidence > OCR_CONFIDENCE_THRESHOLD else 1.0)

        # Draw bounding box for plates
        color = (0, 255, 0) if confidence > 0.5 else (255, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        box_height = 100
        white_box_y1 = max(0, y1 - box_height - 10)
        white_box_y2 = white_box_y1 + box_height
        cv2.rectangle(frame, (x1, white_box_y1), (x2, white_box_y2), (255, 255, 255), -1)

        if roi.size != 0:
            resized_plate = cv2.resize(roi, (x2 - x1, box_height - 30))
            frame[white_box_y1 + 5:white_box_y1 + 5 + resized_plate.shape[0], x1:x1 + resized_plate.shape[1]] = resized_plate

        if plate_text != "":
            cv2.putText(frame, plate_text, (x1 + 5, white_box_y1 + box_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # --- Helmet Detection ---
    helmet_results = helmet_model(frame)[0]

    for helmet_box in helmet_results.boxes:
        x1_h, y1_h, x2_h, y2_h = map(int, helmet_box.xyxy[0])

        if helmet_box.conf[0] < 0.2:
            continue

        class_id = int(helmet_box.cls[0])

        if class_id == 0:
            color = (0, 255, 0)  # Green for helmet
            label = "Helmet"
        else:
            color = (0, 0, 255)  # Red for no helmet
            label = "No Helmet"
            try:
                roi_helmet = frame[y1_h:y2_h, x1_h:x2_h]
                cv2.imwrite(f"output/violations/no_helmet_frame{frame_count}.png", roi_helmet)
            except PermissionError as e:
                print(f"PermissionError saving no helmet frame {frame_count}. Details: {e}")
            violation_writer.writerow([frame_count, "No Helmet", "Helmet not detected"])

        cv2.rectangle(frame, (x1_h, y1_h), (x2_h, y2_h), color, 2)
        cv2.putText(frame, label, (x1_h, y1_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    cv2.imshow("License Plate and Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
out.release()
violation_log.close()
cv2.destroyAllWindows()

# --- Evaluation Metrics ---
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

if successful_detections > 0:
    avg_accuracy = (total_confidence / successful_detections) * 100
    print(f"\nTotal detections: {total_detections}")
    print(f"Successful OCRs: {successful_detections}")
    print(f"Average OCR Accuracy: {avg_accuracy:.2f}%")
else:
    print("No successful OCR detections.")

print("Finished. Output saved to 'output/output_with_text.avi'")

# --- Accuracy/Loss Plotting ---
epoch_size = 10
num_epochs = len(frame_accuracy_list) // epoch_size

training_acc, val_acc = [], []
training_loss, val_loss = [], []

for i in range(num_epochs):
    start = i * epoch_size
    end = start + epoch_size
    epoch_acc = frame_accuracy_list[start:end]
    epoch_loss = frame_loss_list[start:end]

    split = len(epoch_acc) // 2
    training_acc.append(np.mean(epoch_acc[:split]))
    val_acc.append(np.mean(epoch_acc[split:]))
    training_loss.append(np.mean(epoch_loss[:split]))
    val_loss.append(np.mean(epoch_loss[split:]))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(training_acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(training_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("output/accuracy_loss_plot.png")
plt.show()

# --- OCR Confidence Histogram ---
plt.figure()
plt.hist(all_confidences, bins=20, range=(0, 1), color='skyblue', edgecolor='black')
plt.title("OCR Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.savefig("output/ocr_confidence_histogram.png")
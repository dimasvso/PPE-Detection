import cv2
from ultralytics import YOLO

model = YOLO('best.pt') 

video_path = 'video_test.mp4' 
cap = cv2.VideoCapture(video_path)

print("Memulai Deteksi AI... Tekan 'q' untuk berhenti.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, conf=0.5, device=0, imgsz=640)

    annotated_frame = results[0].plot()

    violation_count = 0
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if model.names[class_id] == 'no-helmet':
            violation_count += 1

    # Dashboard Overlay
    cv2.rectangle(annotated_frame, (10, 10), (350, 70), (0, 0, 0), -1)
    color = (0, 255, 0) if violation_count == 0 else (0, 0, 255)
    cv2.putText(annotated_frame, f"PELANGGARAN: {violation_count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Demo CV Safety Check", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
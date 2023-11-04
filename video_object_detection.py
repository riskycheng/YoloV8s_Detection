import cv2
from cap_from_youtube import cap_from_youtube

from yolov8 import YOLOv8

cap = cv2.VideoCapture('rtsp://127.0.0.1:554/stream')
start_time = 5  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

model_path = "./models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

video_fps = cap.get(cv2.CAP_PROP_FPS)

# tune this value below
frames_execute_per_second = 5

# private skipping frequency
skip_freq = int(video_fps / frames_execute_per_second)

print('video fps :' + str(video_fps) + ' skip_freq : ' + str(skip_freq))

frameIndex = 0
executedFrameCount = 0
while cap.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        frameIndex += 1
        if frameIndex % skip_freq != 0:
            continue
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)
    executedFrameCount += 1
    print('executed frames:' + str(executedFrameCount))
    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)
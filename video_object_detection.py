import cv2
from cap_from_youtube import cap_from_youtube
from ConfigManager import ConfigManager
from yolov8 import YOLOv8

min_compute_queue_length = 20
min_scale_factor = 0.6
conf_threshold = 0.3
# clear the global queue when continuous frames are empty
clear_global_queue_reaching_empty_det_length = 15
global_current_continuous_empty_count = 0
global_queue = []


def renderCounter(frame):
    global min_compute_queue_length, min_scale_factor
    frame_height, frame_width = frame.shape[:2]
    cv2.rectangle(
        frame, (0, frame_height - 80), (frame_width, frame_height), (0, 153, 255), -1
    )
    cv2.rectangle(
        frame, (0, frame_height - 80), (frame_width, frame_height), (0, 120, 255), 2
    )
    textAnchor = (int(frame_width / 2) - 120, int(frame_height - 30))
    # check the queue
    if len(global_queue) >= min_compute_queue_length:
        class_full_cnt = 0
        class_empty_cnt = 0
        for item in global_queue:
            if item == 0:
                class_full_cnt += 1
            else:
                class_empty_cnt += 1
        if class_full_cnt >= min_compute_queue_length * min_scale_factor:
            cv2.putText(
                frame,
                "truck full",
                textAnchor,
                cv2.FONT_HERSHEY_COMPLEX,
                1.5,
                (255, 255, 255),
                2,
            )
        elif class_empty_cnt >= min_compute_queue_length * min_scale_factor:
            cv2.putText(
                frame,
                "truck empty",
                textAnchor,
                cv2.FONT_HERSHEY_COMPLEX,
                1.5,
                (255, 255, 255),
                2,
            )


def startExe(rtsp_address):
    global global_current_continuous_empty_count, frames_execute_per_second, \
    clear_global_queue_reaching_empty_det_length, conf_threshold

    cap = cv2.VideoCapture(rtsp_address)
    model_path = "./models/best.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=conf_threshold, iou_thres=0.5)

    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # private skipping frequency
    skip_freq = int(video_fps / frames_execute_per_second)

    skip_freq = max(skip_freq, 1)

    print("video fps :" + str(video_fps) + " skip_freq : " + str(skip_freq))

    frameIndex = 0
    executedFrameCount = 0
    while cap.isOpened():
        # Press key q to stop
        if cv2.waitKey(1) == ord("q"):
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
        print("executed frames:" + str(executedFrameCount))

        if len(boxes) == 0:
            global_current_continuous_empty_count += 1
            if (
                global_current_continuous_empty_count
                >= clear_global_queue_reaching_empty_det_length
            ):
                global_queue.clear()
        else:
            global_current_continuous_empty_count = 0

        # compute the counter
        for box, score, class_id in zip(boxes, scores, class_ids):
            global_queue.append(class_id)

        combined_img = yolov8_detector.draw_detections(frame)

        # render the bottom rectangle
        renderCounter(combined_img)

        cv2.imshow("Detected Objects", combined_img)


if "__main__" == __name__:
    config_manager = ConfigManager("config.json")

    # get the configs
    rtsp_address = config_manager.get("rtsp_address")
    frames_execute_per_second = config_manager.get("frames_execute_per_second")
    min_compute_queue_length = config_manager.get("min_compute_queue_length")

    min_scale_factor = config_manager.get("min_scale_factor")
    clear_global_queue_reaching_empty_det_length = config_manager.get(
        "clear_global_queue_reaching_empty_det_length"
    )
    conf_threshold = config_manager.get("conf_threshold")

    print('Configurated: \n',
          '\trtsp_address:%s \n' %rtsp_address,
          '\tframes_execute_per_second:%d\n' %frames_execute_per_second,
          '\tmin_compute_queue_length:%d \n' %min_compute_queue_length,
          '\tmin_scale_factor:%.2f \n' %min_scale_factor,
          '\tclear_global_queue_reaching_empty_det_length:%d \n' %clear_global_queue_reaching_empty_det_length,
          '\tconf_threshold:%.2f \n' %conf_threshold,
          )


    startExe(rtsp_address)

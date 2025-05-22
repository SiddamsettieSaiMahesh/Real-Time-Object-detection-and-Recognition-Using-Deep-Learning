import cv2
import numpy as np
from utils import draw_boxes

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def detect_objects(video_source):
    net = cv2.dnn.readNetFromCaffe("models/mobilenet_ssd/deploy.prototxt",
                                   "models/mobilenet_ssd/deploy.caffemodel")

    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        frame = draw_boxes(frame, detections, 0.5, CLASSES, COLORS)

        cv2.imshow("Real-Time Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
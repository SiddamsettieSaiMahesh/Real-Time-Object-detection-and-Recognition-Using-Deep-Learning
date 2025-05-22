import cv2

def init_tracker():
    return cv2.TrackerCSRT_create()

def track_objects(video_source):
    cap = cv2.VideoCapture(video_source)
    tracker = None
    initBB = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if initBB is not None:
            success, box = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            initBB = cv2.selectROI("Tracking", frame, fromCenter=False, showCrosshair=True)
            tracker = init_tracker()
            tracker.init(frame, initBB)

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
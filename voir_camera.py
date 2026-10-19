import cv2

pipeline = (
    "udpsrc port=5600 ! application/x-rtp,payload=96,encoding-name=H264 ! "
    "rtpjitterbuffer mode=1 ! rtph264depay ! h264parse ! decodebin ! "
    "videoconvert ! appsink sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Impossible d'ouvrir le flux video.")
else:
    print("Flux video ouvert (appuie sur 'q' pour quitter)...")
    cv2.namedWindow("Camera du drone", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Aucune image recue.")
            break
        cv2.imshow("Camera du drone", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

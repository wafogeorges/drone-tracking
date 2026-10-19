import cv2
import numpy as np

pipeline = (
    "udpsrc port=5600 ! application/x-rtp,payload=96,encoding-name=H264 ! "
    "rtpjitterbuffer mode=1 ! rtph264depay ! h264parse ! decodebin ! "
    "videoconvert ! appsink sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

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

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 100:
                x, y, w, h = cv2.boundingRect(c)
                cx, cy = x + w // 2, y + h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                print(f"Cible detectee en ({cx}, {cy})")

        cv2.imshow("Camera du drone", frame)
        #cv2.imshow("Masque", mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

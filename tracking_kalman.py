import cv2
import numpy as np

pipeline = (
    "udpsrc port=5600 ! application/x-rtp,payload=96,encoding-name=H264 ! "
    "rtpjitterbuffer mode=1 ! rtph264depay ! h264parse ! decodebin ! "
    "videoconvert ! appsink sync=false"
)

lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Detection d'une bbox sous-dimensionnee : toutes les N_VERIF_ETENDUE frames,
# on refait une detection de contour dans une zone elargie autour de la bbox
# CSRT (MARGE_LOCALE x sa taille). Si la vraie tache rouge trouvee est plus
# grande que FACTEUR_CROISSANCE x la bbox actuelle, CSRT est reste coince sur
# un sous-morceau de la cible (typique d'une cible de couleur uniforme, ou
# CSRT n'a aucune raison interne d'agrandir son cadre) : on reinitialise sur
# la vraie taille trouvee.
N_VERIF_ETENDUE = 5
MARGE_LOCALE = 2.0
FACTEUR_CROISSANCE = 1.3


def detecter_cible(frame):
    """Detection HSV : renvoie une bbox (x, y, w, h) ou None si rien trouve."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:
        return None

    return cv2.boundingRect(c)


def detecter_etendue_locale(frame, bbox, marge=MARGE_LOCALE):
    """Cherche la vraie tache rouge dans une zone elargie autour de bbox.

    Renvoie une nouvelle bbox (x, y, w, h) en coordonnees de l'image complete
    si un contour rouge est trouve dans la zone, sinon None.
    """
    x, y, w, h = [int(v) for v in bbox]
    hauteur_frame, largeur_frame = frame.shape[:2]

    cx, cy = x + w / 2, y + h / 2
    demi_largeur, demi_hauteur = (w * marge) / 2, (h * marge) / 2

    x0 = int(max(cx - demi_largeur, 0))
    y0 = int(max(cy - demi_hauteur, 0))
    x1 = int(min(cx + demi_largeur, largeur_frame))
    y1 = int(min(cy + demi_hauteur, hauteur_frame))
    if x1 <= x0 or y1 <= y0:
        return None

    crop = frame[y0:y1, x0:x1]
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv_crop, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_crop, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 100:
        return None

    x_local, y_local, w_local, h_local = cv2.boundingRect(c)
    # On reconvertit les coordonnees du crop local vers l'image complete.
    return (x0 + x_local, y0 + y_local, w_local, h_local)


def creer_kalman(cx, cy):
    """Cree un filtre de Kalman 2D.

    Etat interne (ce que le filtre garde en memoire) : [x, y, vx, vy]
    Mesure fournie a chaque frame (ce qu'on lui donne) : [x, y] seulement

    Modele de mouvement suppose : vitesse constante, donc
        x_suivant = x + vx
        y_suivant = y + vy
    """
    kf = cv2.KalmanFilter(4, 2)

    # measurementMatrix : dit au filtre comment lire une mesure [x, y]
    # a partir de son etat [x, y, vx, vy] -> on ne mesure jamais vx, vy
    # directement, seulement x et y.
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    # transitionMatrix : le modele de mouvement vitesse constante.
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    # processNoiseCov : incertitude sur le modele de mouvement lui-meme
    # (la cible ne va pas vraiment a vitesse parfaitement constante).
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

    # measurementNoiseCov : incertitude sur les mesures HSV/CSRT (le bruit
    # de detection). Plus il est grand, plus le filtre lisse fort et fait
    # moins confiance a chaque mesure individuelle.
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

    kf.errorCovPost = np.eye(4, dtype=np.float32)

    # Etat initial : position de depart connue, vitesse inconnue -> 0.
    kf.statePost = np.array([cx, cy, 0, 0], dtype=np.float32)

    return kf


cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Impossible d'ouvrir le flux video.")
else:
    print("Flux video ouvert (appuie sur 'q' pour quitter)...")
    cv2.namedWindow("Camera du drone", cv2.WINDOW_NORMAL)

    tracker = None
    bbox = None
    kalman = None
    compteur_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Aucune image recue.")
            break

        if tracker is None:
            # Pas de tracker actif : on cherche la cible par couleur.
            bbox = detecter_cible(frame)
            if bbox is not None:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, bbox)
                x, y, w, h = bbox
                kalman = creer_kalman(x + w / 2, y + h / 2)
                compteur_frames = 0
                print("Cible acquise, tracker CSRT + Kalman initialises.")
        else:
            # Tracker actif : on lui demande de suivre la cible.
            success, bbox = tracker.update(frame)
            if not success:
                print("Cible perdue par CSRT, retour a la detection HSV.")
                tracker = None
                bbox = None
            else:
                compteur_frames += 1

                # Verification periodique : la bbox CSRT est-elle restee
                # coincee sur un sous-morceau de la cible ? Si oui, on
                # reinitialise sur la vraie taille avant de nourrir le Kalman,
                # sinon celui-ci lisserait une mesure durablement fausse.
                if compteur_frames % N_VERIF_ETENDUE == 0:
                    x, y, w, h = [int(v) for v in bbox]
                    aire_courante = w * h

                    bbox_locale = detecter_etendue_locale(frame, bbox)
                    if bbox_locale is not None:
                        _, _, w_local, h_local = bbox_locale
                        aire_locale = w_local * h_local

                        if aire_locale > FACTEUR_CROISSANCE * aire_courante:
                            print(f"Bbox sous-dimensionnee detectee "
                                  f"(aire reelle={aire_locale} vs bbox={aire_courante}), "
                                  f"reinitialisation du tracker.")
                            tracker = cv2.TrackerCSRT_create()
                            tracker.init(frame, bbox_locale)
                            bbox = bbox_locale

        # Etape 1 : PREDICTION. Se fait a chaque frame, meme sans mesure --
        # c'est ce qui permet d'avoir une estimation meme quand bbox est None.
        prediction = None
        if kalman is not None:
            prediction = kalman.predict()

        # Etape 2 : CORRECTION. Seulement si on a une mesure cette frame.
        if bbox is not None and kalman is not None:
            x, y, w, h = [int(v) for v in bbox]
            mesure = np.array([[np.float32(x + w / 2)], [np.float32(y + h / 2)]])
            kalman.correct(mesure)

        # Affichage : mesure brute en bleu, estimation Kalman en jaune,
        # pour bien voir visuellement la difference / le lissage.
        if bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            cx, cy = x + w // 2, y + h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # mesure brute (bleu)

        if prediction is not None:
            px, py, vx, vy = prediction.flatten()
            cv2.circle(frame, (int(px), int(py)), 5, (0, 255, 255), -1)  # Kalman (jaune)
            print(f"Kalman: pos=({px:.0f},{py:.0f}) vitesse=({vx:.1f},{vy:.1f}) px/frame")

        cv2.imshow("Camera du drone", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

import cv2
import pygame

# ---------------- SOUND SETUP ----------------
pygame.mixer.init()
pygame.mixer.music.load(r"D:\DriverSenseAI\alarm.wav")

# ---------------- LOAD CASCADES ----------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
print("Camera started")

# ---------------- VARIABLES ----------------
closed_frames = 0
CLOSED_THRESHOLD = 40   # ~2 seconds
alarm_playing = False

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_closed = True

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

        if len(eyes) > 0:
            eyes_closed = False
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    roi_color,
                    (ex, ey),
                    (ex+ew, ey+eh),
                    (0, 255, 0), 2
                )

    # ---------------- LOGIC ----------------
    if eyes_closed:
        closed_frames += 1

        cv2.putText(frame, "EYES CLOSED",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        if closed_frames >= CLOSED_THRESHOLD:
            cv2.putText(frame, "DROWSINESS ALERT!",
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)

            if not alarm_playing:
                pygame.mixer.music.play(-1)
                alarm_playing = True
    else:
        closed_frames = 0
        alarm_playing = False
        pygame.mixer.music.stop()

        cv2.putText(frame, "EYES OPEN",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    cv2.imshow("DriverSenseAI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
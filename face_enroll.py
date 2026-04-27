import cv2
import os
from face_database import FaceDatabase

def enroll():
    name = input("Enter the name of the person to enroll: ")
    db = FaceDatabase()
    
    # Path where raw images will be stored
    save_path = os.path.join(db.db_path, "raw_faces", name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cam = cv2.VideoCapture(0)
    # Using Haar Cascade for initial detection
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    print("Look at the camera. Capturing 30 samples...")

    while count < 30:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            # Crop the face and save it
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{save_path}/{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Face Enrollment', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Enrollment complete for {name}.")

if __name__ == "__main__":
    enroll()
import cv2
import os
import numpy as np
from face_database import FaceDatabase

def trainFromDatabase(db):
    """Train the model from all enrolled faces in database/raw_faces/"""
    raw_faces_path = os.path.join(db.db_path, "raw_faces")
    
    if not os.path.exists(raw_faces_path):
        print("No raw_faces directory found. Cannot train.")
        return False
    
    names = os.listdir(raw_faces_path)
    if len(names) == 0:
        print("No people enrolled yet. Cannot train.")
        return False
    
    #collect all face images and labels
    images = []
    labels = []
    
    #create name-to-label mapping
    names_labels = {name: i for i, name in enumerate(sorted(names))}
    labels_names = {str(label): name for name, label in names_labels.items()}
    
    print(f"Training model for {len(names)} people: {sorted(names)}")
    
    #load all images
    for name in names:
        person_path = os.path.join(raw_faces_path, name)
        if not os.path.isdir(person_path):
            continue
            
        label = names_labels[name]
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)
    
    if len(images) == 0:
        print("No valid images found. Cannot train.")
        return False
    
    print(f"Training on {len(images)} images")
    
    #train LBPH model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    
    #save model and mapping
    db.save_model(recognizer)
    db.save_mapping(labels_names)
    
    print(f"Training complete. Model saved to {db.model_file}")
    return True

def enroll():
    name = input("Enter the name of the person to enroll: ")
    db = FaceDatabase()
    
    #path where raw images will be stored
    save_path = os.path.join(db.db_path, "raw_faces", name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cam = cv2.VideoCapture(0)
    #using Haar Cascade for initial detection
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    print("Look at the camera. Capturing 30 samples")

    while count < 30:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            #crop the face and save it
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{save_path}/{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Captured: {count}/30", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Face Enrollment', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Enrollment complete for {name}. Captured {count} images.")
    
    #ask if user wants to train model now
    #train_now = input("Train model now? y/n: ").lower()
    #if train_now == 'y':
    #    trainFromDatabase(db)
    #else:
    #    print("Skipping training. Run trainModel.py later to train the model.")
    trainFromDatabase(db)

if __name__ == "__main__":
    enroll()

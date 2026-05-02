import cv2
import os
import numpy as np
from face_database import FaceDatabase

def train_from_database():
    """Train the model from all enrolled faces in database/raw_faces/"""
    db = FaceDatabase()
    raw_faces_path = os.path.join(db.db_path, "raw_faces")
    
    if not os.path.exists(raw_faces_path):
        print("Error: No raw_faces directory found.")
        print(f"Expected path: {raw_faces_path}")
        print("Run face_enroll.py first to enroll people.")
        return False
    
    names = os.listdir(raw_faces_path)
    names = [n for n in names if os.path.isdir(os.path.join(raw_faces_path, n))]
    
    if len(names) == 0:
        print("Error: No people enrolled yet.")
        print("Run face_enroll.py to enroll at least one person.")
        return False
    
    #collect all face images and labels
    images = []
    labels = []
    
    #create name-to-label mapping
    names_labels = {name: i for i, name in enumerate(sorted(names))}
    labels_names = {str(label): name for name, label in names_labels.items()}
    
    print(f"Training model for {len(names)} people: {sorted(names)}")
    
    #load all images
    total_images = 0
    for name in names:
        person_path = os.path.join(raw_faces_path, name)
        if not os.path.isdir(person_path):
            continue
            
        label = names_labels[name]
        person_images = 0
        
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)
                person_images += 1
        
        print(f"  {name}: {person_images} images")
        total_images += person_images
    
    if len(images) == 0:
        print("Error: No valid images found.")
        return False
    
    print(f"\nTraining on {total_images} total images...")
    
    #train LBPH model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    
    #save model and mapping
    db.save_model(recognizer)
    db.save_mapping(labels_names)
    
    print(f"Training complete!")
    print(f"Model saved to: {db.model_file}")
    print(f"Mapping saved to: {db.mapping_file}")
    print(f"\nYou can now run main.py to start face recognition.")
    return True

if __name__ == "__main__":
    train_from_database()

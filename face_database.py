import json
import os
import cv2

class FaceDatabase:
    def __init__(self, db_path="database/"):
        self.db_path = db_path
        self.mapping_file = os.path.join(db_path, "map.json")
        self.model_file = os.path.join(db_path, "trainer.yml")
        
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

    def save_mapping(self, labels_names):
        """Saves the ID-to-Name dictionary to a JSON file."""
        with open(self.mapping_file, 'w') as f:
            json.dump(labels_names, f)

    def load_mapping(self):
        """Loads the ID-to-Name dictionary."""
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, 'r') as f:
                return json.load(f)
        return {}

    def save_model(self, LBP):
        """Saves the trained OpenCV LBP model."""
        LBP.save(self.model_file)

    def load_model(self):
        """Loads the trained model for recognition."""
        LBP = cv2.face.LBPHFaceRecognizer_create()
        if os.path.exists(self.model_file):
            LBP.read(self.model_file)
            return LBP
        return None
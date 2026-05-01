import cv2
import numpy as np

class Person:
    def __init__(self, xy, name, frames):
        self.xy = xy
        self.name = name
        self.frames = frames
        self.confidence = 0

def prepareMatch(familiar_dict):
    # Get key value pairs
    images = []
    names = []

    for key, imgs in familiar_dict.items():
        for img in imgs:
            images.append(img)
            names.append(key)

    # name to labels
    names_labels = {name: i for i, name in enumerate(sorted(familiar_dict.keys()))}
    # labels to names
    labels_names = {label: name for name, label in names_labels.items()}

    # Assign the appropriate label for each name in names
    labels = np.array([names_labels[name] for name in names])

    # train
    # opencv-contrib-python
    LBP = cv2.face.LBPHFaceRecognizer_create()
    LBP.train(images, labels)

    return LBP, labels_names


def faceMatch(person, images, LBP, labels_names, num_matches, confidence):
    name_appearences = {}

    # Run for each image
    for im in images:
        label, distance = LBP.predict(im)
        # Confidence threshold check
        if distance < confidence:
            name = labels_names[label]
            # Add to list if not already a key, increment
            if name not in name_appearences:
                name_appearences[name] = 0
            name_appearences[name] += 1

    # If it matches with nobody, it's still unmatched
    if len(name_appearences) <= 0:
        person.name = ""
        return
    
    # Get count of most frequent match and the name
    most_frequent = max(name_appearences, key=name_appearences.get)
    most_frequent_count = name_appearences[most_frequent]

    # If it meets the threshold, updated it, if not, set as unmatched
    if most_frequent_count < num_matches:
        person.name = ""
    else:
        person.name = most_frequent
        
    return
import cv2
import numpy as np
import sys
import os

from faceMatch import Person
from face_database import FaceDatabase

#build background model from multiple frames
def buildBModel(frames):
    accumulator = np.zeros_like(frames[0], dtype=np.float64) #accumulate frames

    for frame in frames:
        accumulator += frame.astype(np.float64)

    bModel = (accumulator/len(frames)).astype(np.uint8) #average
    return bModel


#detect foreground using background subtraction
def detectFG(frame, bModel, threshold): #fg=foreground
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #gray=grayscale
    bModelGray = cv2.cvtColor(bModel, cv2.COLOR_BGR2GRAY)

    frameGray = cv2.GaussianBlur(frameGray, (5, 5), 1) #small blur to reduce sensitivity
    bModelGray = cv2.GaussianBlur(bModelGray, (5, 5), 1)

    diff = cv2.absdiff(frameGray, bModelGray) #absolute diff

    _, fgMask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY) #threshold

    #clean up mask
    structuralElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, structuralElement) #close holes
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, structuralElement) #remove noise

    #remove small components
    numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(fgMask, connectivity=8)
    cleanMask = np.zeros_like(fgMask)
    for i in range(1, numLabels): #skip background label 0
        if stats[i, cv2.CC_STAT_AREA] >= 500: #min area
            cleanMask[labels == i] = 255

    return cleanMask


#update background model
def updateBackground(frame, bModel, fgMask, changeCounter, updateFrames, threshold):
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bModelGray = cv2.cvtColor(bModel, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(frameGray, bModelGray)
    significantChange = diff > threshold #pixels with big difference

    changeCounter[significantChange] += 1 #increment counter
    changeCounter[~significantChange] = 0 #reset unchanged pixels

    updateMask = changeCounter > updateFrames #pixels stable for long enough

    updatedModel = bModel.copy()
    updatedModel[updateMask] = frame[updateMask] #update those pixels
    return updatedModel, changeCounter

#detect faces using haar cascade
def detectFaces(frame, faceCascade, minFaceSize):
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameGray = cv2.equalizeHist(frameGray) #equalize for better detection

    faces = faceCascade.detectMultiScale(
        frameGray,
        scaleFactor=1.1, #scale increment between checks
        minNeighbors=5, #how many neighbors to confirm detection
        minSize=minFaceSize
    )

    if len(faces) == 0:
        return []

    return faces

#match face against trained model
def matchFace(person, images, recognizer, labelToName, matchThreshold, confidenceThreshold):
    nameAppearances = {}
    nameDistances = {} #track distances for each name

    #run for each image
    for img in images:
        label, distance = recognizer.predict(img)
        #confidence threshold check
        if distance < confidenceThreshold:
            name = labelToName.get(str(label), "Unknown")
            #add to list if not already a key, increment
            if name not in nameAppearances:
                nameAppearances[name] = 0
                nameDistances[name] = []
            nameAppearances[name] += 1
            nameDistances[name].append(distance)

    #if it matches with nobody, it's still unmatched
    if len(nameAppearances) <= 0:
        person.name = ""
        person.confidence = 0
        return 0

    #get count of most frequent match and the name
    mostFrequent = max(nameAppearances, key=nameAppearances.get)
    mostFrequentCount = nameAppearances[mostFrequent]

    #if it meets the threshold, update it, if not, set as unmatched
    if mostFrequentCount < matchThreshold:
        person.name = ""
        return 0
    else:
        person.name = mostFrequent
        #get average distance for this name
        avgDistance = sum(nameDistances[mostFrequent])/len(nameDistances[mostFrequent])
        person.confidence = int(avgDistance)
        return int(avgDistance)

#camera and capture settings
numBFrames = 30 #num of frames for initial background, b=background
bUpdateThreshold = 40 #threshold for background update
bUpdateFrames = 5 #frames needed to update background
minFaceSize = (78, 78) #min face size for detection
faceDistancePercent = 20  # How close faces must be in frame percentage to count a single person
faceBufferSize = 15  # Number of faces to store in each person's face buffer
frameMemory = 400  # Number of frames without adding a new face capture to the same person before it "forgets"
matchThreshold = 3 #number of matches to count as a match for each permutation of dict images and images of tracked person
confidenceThreshold = 80 #lower = stricter matching

#if not os.path.exists('savedFrames'): #prob don't need to save every individual frame but putting this here anyways incase we decide to
#    os.makedirs('savedFrames')
if not os.path.exists('detectedFaces'):
    os.makedirs('detectedFaces')

#setup video source
if len(sys.argv) > 1: #video file passed as argument
    src = sys.argv[1]
    if not os.path.exists(src):
        print(f'Error: File {src} not found')
        sys.exit(1)
else:
    src = 0 #webcam index 0

cap = cv2.VideoCapture(src)

if not cap.isOpened():
    print('Error: Could not open video source')
    sys.exit(1)

print(f'Video source opened: {src}')

#load haar cascade for face detection
faceXml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(faceXml):
    print(f'Error: Haar cascade not found at {faceXml}')
    sys.exit(1)

faceCascade = cv2.CascadeClassifier(faceXml) #face detector

#load face database and trained model
db = FaceDatabase()
recognizer = db.load_model()
labelToName = db.load_mapping()

if recognizer is None:
    print('Error: No trained model found. Run face_enroll.py first to enroll people.')
    print('Or run trainModel.py to train from existing database/raw_faces/')
    sys.exit(1)

print(f'Loaded model with {len(labelToName)} people: {list(labelToName.values())}')

totalFacesDetected = 0
totalFrames = 0

#capture initial background frames
print(f'Making background model from first {numBFrames} frames...')
bFrames = [] #b=background

for i in range(numBFrames):
    ret, frame = cap.read()
    if not ret:
        print(f'Could only capture {i} frames for background')
        break
    bFrames.append(frame)
    print(f'Captured background frame {i+1}/{numBFrames}')

#build initial background model
bModel = buildBModel(bFrames) #b=background model

print('Background model built')

#initialize change counter for background update
changeCounter = np.zeros(bModel.shape[:2], dtype=np.float32) #tracks changes

# Get height and width of frame
fHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

# Get distances from percents
face_distance_x = faceDistancePercent * .01 * fWidth
face_distance_y = faceDistancePercent * .01 * fHeight

# Map of frames for people
people_info = []  # Person objects
people_dict = {}  # Key is a Person, value is list of images

#find next instance number by checking all person folders
instanceNum = 1
if os.path.exists('detectedFaces'):
    for personFolder in os.listdir('detectedFaces'):
        personPath = os.path.join('detectedFaces', personFolder)
        if os.path.isdir(personPath):
            for instance in os.listdir(personPath):
                instancePath = os.path.join(personPath, instance)
                if os.path.isdir(instancePath) and instance.isdigit():
                    instanceNum = max(instanceNum, int(instance) + 1)

print(f'Instance number for this run: {instanceNum}')

#main loop
print('Starting detection, press Q to quit')
while True:
    ret, cFrame = cap.read() #c=current

    if not ret:
        print('End of video or camera disconnected')
        break

    totalFrames+=1

    for person in people_info:
        person.frames += 1

    # Remove people if mem is gone
    remove_people = [person for person in people_info if person.frames >= frameMemory]
    for person in remove_people:
        del people_dict[person]
        people_info.remove(person)

    #detect foreground
    fgMask = detectFG(cFrame, bModel, bUpdateThreshold) #fg=foreground

    #update background model
    bModel, changeCounter = updateBackground(cFrame, bModel, fgMask, changeCounter, bUpdateFrames, bUpdateThreshold)

    #detect faces in current frame
    faces = detectFaces(cFrame, faceCascade, minFaceSize)

    #draw foreground mask and bounding boxes on display frame
    displayFrame = cFrame.copy()

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            #check if face overlaps with foreground mask, so we only get moving people, we will see if we change this or not
            faceMask = fgMask[y:y+h, x:x+w]
            fgRatio = np.sum(faceMask > 0)/(w*h) #fg=foreground ratio
            # Looks at x and y and determine the closest face to determine which existing Person to associate it with, or a new one
            coords = (x, y)
            closest_person = min(people_info, key=lambda l: (l.xy[0] - coords[0]) ** 2 + (l.xy[1] - coords[1]) ** 2,
                                 default=None)
            if closest_person is not None:
                closest_coords = closest_person.xy
                if abs(coords[0] - closest_coords[0]) <= face_distance_x and abs(
                        coords[1] - closest_coords[1]) <= face_distance_y:
                    # Remove and update old coordinate value and frame count
                    current_person = closest_person
                    current_person.xy = coords
                    current_person.frames = 0
                else:
                    # New person object at coordinates
                    current_person = Person(coords, "", 0)
                    current_person.confidence = 0
                    people_info.append(current_person)
                    people_dict[current_person] = []
            else:
                current_person = Person(coords, "", 0)
                current_person.confidence = 0
                people_info.append(current_person)
                people_dict[current_person] = []

            if fgRatio > 0.15: #at least 15% of face region is foreground
                totalFacesDetected+=1

                # Draw box around face; Unknowns are red, knowns are green
                if current_person.name != "":
                    color = (0, 255, 0)
                    label = f"{current_person.name} ({current_person.confidence})"
                else:
                    color = (0, 0, 255)
                    label = f"Unknown ({current_person.confidence})"
                cv2.rectangle(displayFrame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(displayFrame, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                #crop face
                faceImg = cFrame[y:y+h, x:x+w] #crop face region

                #save face to appropriate folder with instance number
                personFolder = current_person.name if current_person.name != "" else "Unknown"
                personPath = os.path.join('detectedFaces', personFolder, str(instanceNum))
                if not os.path.exists(personPath):
                    os.makedirs(personPath)
                #add confidence to filename
                conf = getattr(current_person, 'confidence', 0)
                fFilename = os.path.join(personPath, f'face{totalFacesDetected}_frame{totalFrames}_conf{conf}.png') #f=face
                cv2.imwrite(fFilename, faceImg)
                # Instead of saving to file, put them in a list
                # Each person-list is compared against a face dictionary using LBPs
                # Testing this the cv2 built-in way as a proof of concept
                people_dict[current_person].append(cv2.resize(cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY), (200, 200)))
                if len(people_dict[current_person]) > faceBufferSize:
                    people_dict[current_person].pop(0)
                # Don't match if there are less than 3 face captures
                if len(people_dict[current_person]) >= 3:
                    matchFace(current_person, people_dict[current_person], recognizer, labelToName, matchThreshold, confidenceThreshold)

                print(f'Frame {totalFrames}: Face #{totalFacesDetected} detected at x={x}, y={y}, size={w}x{h}, name={current_person.name if current_person.name else "Unknown"}, confidence={current_person.confidence}')

    #show fg mask as green overlay
    fgColored = np.zeros_like(cFrame)
    fgColored[:, :, 1] = fgMask #green channel

    combined = cv2.addWeighted(displayFrame, 1, fgColored, 0.5, 0) #blend mask onto frame

    #frame info
    cv2.putText(combined, f'Frame: {totalFrames}  Faces: {totalFacesDetected}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
    cv2.putText(displayFrame, f'Frame: {totalFrames}  Faces: {totalFacesDetected}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)

    cv2.imshow('Background Model', bModel)
    cv2.imshow('Foreground Mask', fgMask)
    cv2.imshow('Basic', displayFrame)
    cv2.imshow('Detection', combined)

    #save frame, only used if we decide to use it
    #sFilename = f'savedFrames/frame{totalFrames}.png' #s=saved
    #cv2.imwrite(sFilename, cFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Quit key pressed')
        break

#cleanup
cap.release()
cv2.destroyAllWindows()
print(f'Done. Processed {totalFrames} frames, detected {totalFacesDetected} faces total')
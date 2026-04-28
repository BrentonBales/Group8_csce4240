import cv2
import numpy as np
import sys
import os
import math

from faceMatch import prepareMatch, faceMatch, Person

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

#camera and capture settings
numBFrames = 30 #num of frames for initial background, b=background
bUpdateThreshold = 40 #threshold for background update
bUpdateFrames = 5 #frames needed to update background
minFaceSize = (60, 60) #min face size for detection
faceDistancePercent = 7  # How close faces must be in frame percentage to count a single person
faceBufferSize = 15  # Number of faces to store in each person's face buffer
frameMemory = 1000  # Number of frames without adding a new face capture to the same person before it "forgets"
match_threshold = 3  # Number of matches to count as a match for each permutation of dict images and images of tracked person

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

#show background model
#cv2.imshow('Background Model', bModel) #we prob don't need to show this rn, since its display isn't updating
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

# Get faces from dict
familiar_faces = {}

names = os.listdir(os.path.join(".", "dict"))

# Get dict images in correct size and grayscale
for n in names:
    if n not in familiar_faces:
        familiar_faces[n] = []
    for file in os.listdir(os.path.join(".", "dict", n)):
        im = cv2.imread(os.path.join(".", "dict", n, file))
        familiar_faces[n].append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (200, 200)))

# Get recognizer and the label-name conversion
rec, l2n = prepareMatch(familiar_faces)

#main loop
print('Starting detection, press Q to quit')
while True:
    ret, cFrame = cap.read() #c=current

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
                    people_info.append(current_person)
                    people_dict[current_person] = []
            else:
                current_person = Person(coords, "", 0)
                people_info.append(current_person)
                people_dict[current_person] = []

            if fgRatio > 0.2: #at least 20% of face region is foreground
                totalFacesDetected+=1

                # Draw box around face; Unknowns are red, knowns are green
                if current_person.name != "":
                    color = (0, 255, 0)
                    label = current_person.name
                else:
                    color = (0, 0, 255)
                    label = "Unknown"
                cv2.rectangle(displayFrame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(displayFrame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                #crop face
                faceImg = cFrame[y:y+h, x:x+w] #crop face region

                # Instead of saving to file, put them in a list
                # Each person-list is compared against a face dictionary using LBPs
                # Testing this the cv2 built-in way as a proof of concept
                people_dict[current_person].append(cv2.resize(cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY), (200, 200)))
                if len(people_dict[current_person]) > faceBufferSize:
                    people_dict[current_person].pop(0)
                # Don't match if there are less than 3 face captures
                if len(people_dict[current_person]) >= 3:
                    faceMatch(current_person, people_dict[current_person], rec, l2n, match_threshold, 60)

                print(f'Frame {totalFrames}: Face #{totalFacesDetected} detected at x={x}, y={y}, size={w}x{h}')

    #show fg mask as green overlay
    fgColored = np.zeros_like(cFrame)
    fgColored[:, :, 1] = fgMask #green channel

    combined = cv2.addWeighted(displayFrame, 1, fgColored, 0.5, 0) #blend mask onto frame

    #frame info
    cv2.putText(combined, f'Frame: {totalFrames}  Faces: {totalFacesDetected}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2)

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
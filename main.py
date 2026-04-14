import cv2
import numpy as np
import sys
import os

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

#camera and capture settings
numBFrames = 30 #num of frames for initial background, b=background
bUpdateThreshold = 40 #threshold for background update
bUpdateFrames = 5 #frames needed to update background

#setup video source
if len(sys.argv) > 1: #video file passed as argument
    src = sys.argv[1]
    if not os.path.exists(src):
        print(f'Error: File {src} not found')
        sys.exit(1)
else:
    src = 0 #webcam index 0

cap = cv2.VideoCapture(src)

#load haar cascade for face detection
faceXml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(faceXml):
    print(f'Error: Haar cascade not found at {faceXml}')
    sys.exit(1)

#capture initial background frames
print(f'Making background model from first {numBFrames} frames...')
bFrames = [] #b=background

for i in range(numBFrames):
    ret, frame = cap.read()
    bFrames.append(frame)
    print(f'Captured background frame {i+1}/{numBFrames}')

#build initial background model
bModel = buildBModel(bFrames) #b=background model

#show background model
cv2.imshow('Background Model', bModel)
print('Background model built')

#initialize change counter for background update
changeCounter = np.zeros(bModel.shape[:2], dtype=np.float32) #tracks changes

#main loop
print('Starting detection, press Q to quit')
while True:
    ret, cFrame = cap.read() #c=current

    #detect foreground
    fgMask = detectFG(cFrame, bModel, bUpdateThreshold) #fg=foreground

    #update background model
    bModel, changeCounter = updateBackground(cFrame, bModel, fgMask, changeCounter, bUpdateFrames, bUpdateThreshold)

    cv2.imshow('Foreground Mask', fgMask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Quit key pressed')
        break

#cleanup
cap.release()
cv2.destroyAllWindows()
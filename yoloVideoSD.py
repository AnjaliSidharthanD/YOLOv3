##### SCRIPT STARTS HERE #####
#!usr/bin/bash python
# Importing required packages
#!python3 yoloVideoSD.py --video=/content/gdrive/MyDrive/yoloV3/PP.mp4

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import re
from scipy.spatial import distance as dist

# Initialize the parameters
confidenceThreshold = 0.3  #Confidence threshold
nmsThreshold = 0.4         #Non-maximum suppression threshold
inputWidth = 416           #Width of network's input image
inputHeight = 416          #Height of network's input image
minDistance = 50          # define the minimum safe distance (in pixels) that two people can be from each other

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help="True/False", default=False)
parser.add_argument('--video', help="Path to video file", default="videos/car_on_road.mp4")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()

#Load YOLO V3
def loadYolo():

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")

    # derive the paths to the YOLO weights and model configuration
    configPath = '/content/gdrive/MyDrive/yoloV3/files/yolov3.cfg'
    weightsPath = '/content/gdrive/MyDrive/yoloV3/files/yolov3.weights'

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load the COCO class labels our YOLO model was trained on
    classes=[]
    # load the COCO class labels our YOLO model was trained on
    classesPath = '/content/drive/MyDrive/yoloV3/files/coco.names'
    with open(classesPath, "r") as f:
      classes = [line.strip() for line in f.readlines()]

    # initialize a list of colors to represent each possible class label
    np.random.seed(4)
    colors = np.random.randint(0, 255, size=(len(classes), 3),dtype="uint8")
    
    # and determine only the *output* layer names that we need from YOLO
    
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    outputLayers = [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print("[INFO] YOLO V3 loaded successfully")
    return net, classes, colors, outputLayers

def detectObjects(image,net,outputLayers):
    # Create a 4D blob from a frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, size=(inputWidth, inputHeight), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(outputLayers)
    return blob, layerOutputs

def getBoundingbox(layerOutputs, height, width, classes):
    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    centroids = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        #loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            #filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability

            if confidence > confidenceThreshold and str(classes[classID]) == 'person':
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, bbWidth, bbHeight) = box.astype("int")
                
                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (bbWidth / 2))
                y = int(centerY - (bbHeight / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(bbWidth), int(bbHeight)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                centroids.append((centerX,centerY))

    return boxes, confidences, classIDs, centroids


# Draw the predicted bounding boxes

#def drawPredictedBB(boxes, confidences, colors, classIDs,classes, image):
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes with lower confidences
 #   indices = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold)
 #   font = cv2.FONT_HERSHEY_PLAIN
    
    # ensure at least one detection exists
 #   if len(indices) > 0:
        #loop over indices we are keeping
 #       for index in indices.flatten():
            # extract the bounding box coordinates
 #           (x, y) = (boxes[index][0], boxes[index][1])
 #           (width,height) = (boxes[index][2], boxes[index][3])

            # Draw a bounding box
 #           text = "{}: {:.4f}".format(classes[classIDs[index]],confidences[index])
 #           color = [int(c) for c in colors[classIDs[index]]]
 #           cv2.rectangle(image, (x,y), (x+width, y+height), color,2)
 #           cv2.putText(image, text, (x, y - 5), font, 1, color, 1)
 #
def videoToFrames(videoPath): 
    print("[INFO] video to frame conversion started")
    frameCount=0
    inputFrames=[]
    capture = cv2.VideoCapture(videoPath)
    while(True):
        # Capture the video frame by frame from the file
        (grabbed,frame) = capture.read()
        # if the frame was not grabbed, then we have reached the end
        #of the stream
        if not grabbed:
          print("[INFO] All frames appended !!!")
          break
        inputFrames.append(frame)
        frameCount = frameCount + 1
        cv2.imwrite('inputFrames/'+str(frameCount)+'.png', frame)
    print("[INFO] inputFrames directory formed successfully")
    return inputFrames 

def framesToVideo():
  print("[INFO] stitching frames to video started")
  outputFrames = os.listdir('output/')
  outputFrames.sort(key=lambda f: int(re.sub('\D', '', f)))
  frames=[]
  writer = None
  (width,height)= (None, None)
  for index in range(len(outputFrames)):
    #reading each files
    image = cv2.imread('output/'+outputFrames[index])
    height, width = image.shape[:2]
    size = (width,height)
    
    #inserting the frames into an image array
    frames.append(image)
  
  size = (width,height)
  outputFile = videoPath[:-4]+'_yolo_out_py.mp4'
  # initialize our video writer
  fourcc = cv2.VideoWriter_fourcc(*'DIVX')
  writer = cv2.VideoWriter(outputFile,fourcc, 29, size)

  for index in range(len(frames)):
    # writing to a image array
         writer.write(frames[index])
  writer.release()
  print("[INFO] Frames are stitched to video successfully")


def personDetection(inputFrames):
  net, classes, colors, outputLayers = loadYolo()
  print("[INFO] person detection started")
  for i in range(len(inputFrames)):
    frame = cv2.imread('inputFrames/'+str(i+1)+'.png')
    height,width = frame.shape[:2]
    # construct a blob from the input frame and then perform a forward
	  # pass of the YOLO object detector, giving us our bounding boxes
	  # and associated probabilities
    blob, layerOutputs = detectObjects(frame, net, outputLayers)
    boxes, confidences, classIDs = getBoundingbox(layerOutputs, height, width,classes)
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes with lower confidences
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold)
    font = cv2.FONT_HERSHEY_PLAIN

    results =[]

    if len(indices) > 0:
      for index in indices.flatten():
        # extract the bounding box coordinates
        (x,y) = (boxes[index][0], boxes[index][1])
        (w,h) = (boxes[index][2], boxes[index][3])
        results.append((confidences[index], (x, y, x + w, y + h), centroids[index]))

      
      violate =()
      if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")
        for i in range(0, D.shape[0]) :
          for j in range(i + 1, D.shape[1]) :
            if D[i, j] < minDistance :
              violate.add(i)
              violate.add(j)

      for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        if i in violate :
          color = (0, 0, 255)
        
        cv2.rectangle(frame,(startX, startY),(endX, endY),color,2)
        cv2.circle(frame, (cX, cY), 5, (255,0,0), 1)

      # show the output frame
      #text = "Social Distancing Violations: {}".format(len(violate))
	    #cv2.putText(frame, text, (10, frame.shape[0] - 25),
      #           cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
      cv2.imwrite('output/'+str(i+1)+'.png', frame)

videoPath = args.video
inputFrames= videoToFrames(videoPath)
person = personDetection(videoPath)
framesToVideo()

# Closes all the frames
cv2.destroyAllWindows()
   
print("The video was successfully saved")
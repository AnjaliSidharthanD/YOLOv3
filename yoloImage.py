%%writefile yoloImage.py
##### SCRIPT STARTS HERE #####
#!usr/bin/bash python
# Importing required packages
import numpy as np
import argparse
import time
import cv2
from scipy.spatial import distance as dist
from google.colab.patches import cv2_imshow


#initialize parameters
inputPath="people.jpg"
outputPathDetection = inputPath[:-4]+'OutDetection.png'
outputPathSD = inputPath[:-4]+'OutSD.png'
classesPath='files/coco.names'
configurationPath='files/yolov3.cfg'
weightsPath='files/yolov3.weights'
confidenceThreshold=0.3
nmsThreshold =0.3
minDistance =300

classes = []
with open(classesPath, "r") as f:
  classes = [line.strip() for line in f.readlines()]

np.random.seed(47)
COLORS = np.random.randint(0, 255, size=(len(classes), 3),
	dtype="uint8")

net = cv2.dnn.readNetFromDarknet(configurationPath , weightsPath)

image = cv2.imread(inputPath)
(height,width) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []
centroids = []

# loop over each of the layer outputs
for output in layerOutputs:
  # loop over each of the detections
  for detection in output:
    # extract the class ID and confidence (i.e., probability) of
    # the current object detection
    scores = detection[5:]
    classID = np.argmax(scores)
    confidence = scores[classID]
    # filter out weak predictions by ensuring the detected
    # probability is greater than the minimum probability
    if confidence > confidenceThreshold and str(classes[classID]) == 'person':
      # scale the bounding box coordinates back relative to the
      # size of the image, keeping in mind that YOLO actually
      # returns the center (x, y)-coordinates of the bounding
      # box followed by the boxes' width and height
      box = detection[0:4] * np.array([width, height, width, height])
      (centerX, centerY, bbWidth, bbHeight) = box.astype("int")
      
      # use the center (x, y)-coordinates to derive the top
      # and and left corner of the bounding box
      x = int(centerX - (bbWidth / 2))
      y = int(centerY - (bbHeight / 2))

      boxes.append([x, y, int(bbWidth), int(bbHeight)])
      confidences.append(float(confidence))
      classIDs.append(classID)
      centroids.append(((centerX), (centerY)))

indices = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold,
	nmsThreshold)
results = []

if len(indices) > 0 :
  for index in indices.flatten():
    (x, y) = (boxes[index][0], boxes[index][1])
    (w, h) = (boxes[index][2], boxes[index][3])
    color = (255, 0, 0)
    cv2.rectangle(image,(x, y),(x+w, y+h),color,2)
    results.append((confidences[index], (x, y, x + w, y + h), centroids[index]))
cv2.imwrite(outputPathDetection, image)

violate = set()
if len(results) >= 2:
  centroids = np.array([r[2] for r in results])
  D = dist.cdist(centroids, centroids, metric="euclidean")
  for i in range(0, D.shape[0]):
    for j in range(i + 1, D.shape[1]):
      if D[i, j] < minDistance:
        violate.add(i)
        violate.add(j)

for (i, (prob, bbox, centroid)) in enumerate(results):
  (startX, startY, endX, endY) = bbox
  (cX, cY) = centroid
  color = (0, 255, 0)

  if i in violate :
    color = (0, 0, 255)
  
  cv2.rectangle(image,(startX, startY),(endX, endY),color,2)
  cv2.circle(image, (cX, cY), 5, (255,0,0), 1)

text = "Social Distancing Violations: {}".format(len(violate))
cv2.putText(image, text, (10, image.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
# show the output image
cv2.imwrite(outputPathSD, image)
print("Link to output : ", outputPathSD)

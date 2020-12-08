# YOLO OBJECT DETECTION OpenCV

# Import the necessary packages
import numpy as np
import argparse
#import time
import cv2
import os
from os import listdir
from os.path import isfile, join

class YOLO_model(object):

	def __init__(self, labels, weights, config):
		self.labelsPath = labels
		self.weightsPath = weights
		self.configPath = config

	# Method to predict from class variables and input image 'myimage'
	def predict(self, myimage):

		# Load the YOLO object detector trained on COCO dataset using 80 classes
		net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath) # BUILDING THE MODEL

		# Load the input image and grab its spatial dimensions
		file_names = myimage
		image = cv2.imread(myimage)
		image = cv2.resize(image, None, fx = 0.3, fy = 0.3)
		(H, W) = image.shape[:2]

		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		# We construct a blob form the input and then perform a forward pass of the YOLO
		# object detector, giving us the bounding boxes and associated probabilities
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
		net.setInput(blob)
		#start = time.time() ######
		layerOutputs = net.forward(ln) # FORWARD PASS THROUGH THE MODEL
		#end = time.time() ######

		# Show timing information on YOLO
		#print("[INFO] YOLO took {:.6f} seconds".format(end - start)) ######

		# We initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		LABELS = open(self.labelsPath).read().strip().split("\n")
		COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype = "uint8")

		# Iterate over each of the layer outputs
		# layerOutputs are the predictions of our model
		for output in layerOutputs:
			# iterate over each of the detections
			for detection in output:
				# Here we extract the class ID and confidence/prob of the current object detection
				scores = detection[5:]
				classID = np.argmax(scores) # it returns the indeces of what we have detected
				confidence = scores[classID]

				# Filter out weak predictions by ensuring the detected probebility is greater than the minimum probability
				if confidence > 0.5:
					# Scale the bounding box coordinates back relative to the size of the image,
					# Keeping in mind that YOLO actually return the center (x, y)-coordinates of the bounding box
					# followed by the boxes' width and height

					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# Use the center (x, y)-coordinates to derive the top and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# Update our List of bounding box coordinates, confidences and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# apply non-maxima suppression to suppres weak overlapping bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

		# List of identified objects in the image/frame
		identified_objects = []

		# We ensure at Least one detection exists
		if len(idxs) > 0: # if one detection exists
			# iterate over the indexes we are keeping
			for i in idxs.flatten():

				identified_objects.append(LABELS[classIDs[i]])

		# Return the list of detected objects:
		# Each repeating object is repeated in the list as well
		return identified_objects
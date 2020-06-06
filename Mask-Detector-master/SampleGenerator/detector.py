from concurrent import futures
import logging
import sys
import os
import dlib
import glob
import grpc
import io
import numpy as np
from PIL import Image
import cv2


def detect_faces(image: np.ndarray, display_image:bool = True):
	detector = dlib.get_frontal_face_detector()
	shape_predictor = dlib.shape_predictor("./trained_models/shape_predictor.dat")

	# Ask the detector to find the bounding boxes of each face. The 1 in the
	# second argument indicates that we should upsample the image 1 time. This
	# will make everything bigger and allow us to detect more faces.
	detected_faces = detector(image, 1)
	# print("Number of faces detected: {}".format(len(detected_faces)))

	detected_faces_boxes = []
	detected_faces_landmarks = []

	# Now process each face we found.
	for k, d in enumerate(detected_faces):
		detected_faces_boxes.append([
			d.left(),
			d.top(),
			d.right(),
			d.bottom()
		])

		# Get the landmarks/parts for the face in box d.
		shape = shape_predictor(image, d)
		shape = shape_to_np(shape)
		detected_faces_landmarks.append(shape)

		if display_image == True:
			for (x, y) in shape:
				cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	if display_image == True:
		cv2.imshow("Output", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
		cv2.waitKey(0)

	return detected_faces_boxes, detected_faces_landmarks


def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
		
	return coords

if __name__ == "__main__":
	img_path = "./test_images/faces.jpg"
	image = np.array(Image.open(img_path).convert("RGB"))
	detect_faces(image)

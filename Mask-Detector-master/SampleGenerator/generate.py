import pathlib
from typing import List, Tuple, Dict 
import os
import numpy as np
from PIL import Image
import cv2
import random
from tqdm import tqdm
import json

from detector import detect_faces

def generate_dataset(root_face_images_folder_path, mask_images_folder = "mask_images"):
	original_images_paths = list(pathlib.Path(root_face_images_folder_path).glob('*'))
	mask_images_paths = list(pathlib.Path(mask_images_folder).glob('*'))
	mask_images = [Image.open(path) for path in mask_images_paths]

	targets = {}

	for i, img_path in enumerate(tqdm(original_images_paths)):
		image_name = str(img_path).split(os.sep)[1]
		image = Image.open(img_path).convert("RGB").resize((800,800))
		detected_faces_boxes, detected_faces_landmarks = detect_faces(np.array(image), False)
		if len(detected_faces_boxes) == 0:
			continue

		image.save("resized_images/" + image_name)
		image = apply_face_mask(image, mask_images, detected_faces_landmarks)
		image.save("generated_images/" + image_name)

		targets[image_name] = {
			"bbox": detected_faces_boxes,
		}

		# cv2.imshow("Output", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
		# cv2.waitKey(0)

	with open("targets.json", "w", encoding="utf8") as outfile:
		json.dump(targets, outfile)


def apply_face_mask(face_image, mask_images, detected_faces_landmarks):
	for idx, face_landmarks in enumerate(detected_faces_landmarks):
		mask_image = random.choice(mask_images)

		nose_point = face_landmarks[28]
		nose_v = np.array(nose_point)

		chin_bottom_point = face_landmarks[9]
		chin_bottom_v = np.array(chin_bottom_point)
		chin_left_point = face_landmarks[6]
		chin_right_point = face_landmarks[12]

		# split mask and resize
		width = mask_image.width
		height = mask_image.height
		width_ratio = 1.7
		new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

		# left
		mask_left_img = mask_image.crop((0, 0, width // 2, height))
		mask_left_width = get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
		mask_left_width = int(mask_left_width * width_ratio)
		mask_left_img = mask_left_img.resize((mask_left_width, new_height))

		# right
		mask_right_img = mask_image.crop((width // 2, 0, width, height))
		mask_right_width = get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
		mask_right_width = int(mask_right_width * width_ratio)
		mask_right_img = mask_right_img.resize((mask_right_width, new_height))

		# merge mask
		size = (mask_left_img.width + mask_right_img.width, new_height)
		mask_image = Image.new('RGBA', size)
		mask_image.paste(mask_left_img, (0, 0), mask_left_img)
		mask_image.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

		# rotate mask
		angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
		rotated_mask_image = mask_image.rotate(angle, expand=True)

		# calculate mask location
		center_x = (nose_point[0] + chin_bottom_point[0]) // 2
		center_y = (nose_point[1] + chin_bottom_point[1]) // 2

		offset = mask_image.width // 2 - mask_left_img.width
		radian = angle * np.pi / 180
		box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_image.width // 2
		box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_image.height // 2

		# add mask
		face_image.paste(rotated_mask_image, (box_x, box_y), rotated_mask_image)
	
	return face_image

def get_distance_from_point_to_line(point, line_point1, line_point2):
	distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
					  (line_point1[0] - line_point2[0]) * point[1] +
					  (line_point2[0] - line_point1[0]) * line_point1[1] +
					  (line_point1[1] - line_point2[1]) * line_point1[0]) / \
			   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
					   (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
	return int(distance)

def main():
	generate_dataset(root_face_images_folder_path="original_images")
	# generate_dataset(root_face_images_folder_path="test_images")

if __name__ == "__main__":
	main()
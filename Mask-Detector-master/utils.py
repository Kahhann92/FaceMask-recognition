import numpy as np
import cv2

def imshow(inp, title=None):
	"""Imshow for Tensor."""
	inp = inp.numpy().transpose((1, 2, 0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	inp = std * inp + mean
	inp = np.clip(inp, 0, 1)
	plt.imshow(inp)
	if title is not None:
		plt.title(title)
	plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, dataloader, num_images=5):
	was_training = model.training
	model.eval()
	images_so_far = 0
	fig = plt.figure()

	with torch.no_grad():
		inputs, labels = next(iter(dataloader))
		inputs = inputs.to(device)
		labels = labels.to(device)

		outputs = model(inputs)
		_, preds = torch.max(outputs, 1)

		for j in range(inputs.size()[0]):
			images_so_far += 1
			ax = plt.subplot(num_images//2, 2, images_so_far)
			ax.axis('off')
			ax.set_title('predicted: {}'.format(class_names[preds[j]]))
			imshow(inputs.cpu().data[j])

			if images_so_far == num_images:
				model.train(mode=was_training)
				return
		model.train(mode=was_training)


def compute_cam(feature_conv, weight_softmax, class_idx):
	# generate the class activation maps upsample to 256x256
	size_upsample = (256, 256)
	bz, nc, h, w = feature_conv.shape
	output_cam = []
	for idx in class_idx:
		cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
		cam = cam.reshape(h, w)
		cam = cam - np.min(cam)
		cam_img = cam / np.max(cam)
		cam_img = np.uint8(255 * cam_img)
		output_cam.append(cv2.resize(cam_img, size_upsample))
	return output_cam


def get_cam(model, features, image_tensor, classes, image_path = "sample.jpg"):
	params = list(model.parameters())
	weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

	output = model(image_tensor.unsqueeze(0))
	output = output.squeeze()

	probs, idx = output.sort(0, True)

	for i in range(0, 2):
		line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
		print(line)


	CAMs = compute_cam(features[0], weight_softmax, [idx[0].item()])

	print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0].item()])
	img = cv2.imread(image_path)
	height, width, _ = img.shape
	CAM = cv2.resize(CAMs[0], (width, height))
	heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
	result = heatmap * 0.3 + img * 0.5
	cv2.imwrite('cam.jpg', result)
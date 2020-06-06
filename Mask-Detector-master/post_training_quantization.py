import torch
import os
print(torch.__version__)

from models import model_rcnn

def apply_quantization(model):
	model.qconfig = torch.quantization.default_qconfig
	torch.quantization.prepare(model, inplace=True)
	torch.quantization.convert(model, inplace=True)

	return model

def print_size_of_model(model):
	torch.save(model.state_dict(), "temp.p")
	print('Size (MB):', os.path.getsize("temp.p")/1e6)
	os.remove('temp.p')

def load_model():
	model = model_rcnn.create_model(3)
	model.load_state_dict(torch.load("model.pt"))
	model.eval()
	return model

def main():	
	model = load_model()

	print("Size of model before quantization")
	print_size_of_model(model)

	model = apply_quantization(model)
	
	print("Size of model after quantization")
	print_size_of_model(model)

	torch.save(model.state_dict(), f"quantized_model.pt")


if __name__ == "__main__":
	main()
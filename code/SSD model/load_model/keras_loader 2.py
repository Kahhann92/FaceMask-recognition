from keras.models import model_from_json

def load_keras_model(json_path, weight_path):
    model = model_from_json(open(json_path).read())
    model.load_weights(weight_path)
    return model


def keras_inference(model, img_arr):
    result = model.predict(img_arr)
    y_bboxes= result[0]
    y_scores = result[1]
    return y_bboxes, y_scores
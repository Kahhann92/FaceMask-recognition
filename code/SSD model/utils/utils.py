from keras.callbacks import Callback
import keras.backend as K
class PrintLearningRate(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print("Current lr is %.9f" % K.eval(self.model.optimizer.lr))

def generate_anchor_config(model, detection_layer_config):
    feature_map_sizes = []
    anchor_ratios = []
    anchor_scales = []

    for layer_name in sorted(detection_layer_config.keys()):
        feature_map_size = model.get_layer(layer_name).output.shape[1:3]
        ratio = detection_layer_config[layer_name]['aspect_ratio']
        scale = detection_layer_config[layer_name]['scale']

        feature_map_sizes.append([int(feature_map_size[0]), int(feature_map_size[1])])
        anchor_ratios.append(ratio)
        anchor_scales.append(scale)
    return {'feature_map_sizes':feature_map_sizes,
            'anchor_scales':anchor_scales,
            'anchor_ratios':anchor_ratios}
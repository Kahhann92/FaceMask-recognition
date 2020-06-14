# -*- encoding=utf-8 -*-
from keras.layers import Input, MaxPool2D
from keras.models import Model
from model_builder.model_builder_util import _conv_bn_relu

def ConvNet(input_shape):
    img_input= Input(input_shape, name='data')
    x = img_input
    # x = ZeroPadding2D(((1, 0), (1, 0)),name='data_pad')(img_input)
    # for idx, filter in enumerate([32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]):
    #     if idx in [0, 1, 3, 6, 8, 10, 11]:

    for idx, filter in enumerate([32, 64, 64, 64, 128, 128, 64, 64]):
        # 在特定卷基层添加池化层
        if idx in [0, 1, 2, 3, 4, 5]:
            add_pool = True
        else:
            add_pool = False
        padding = 'same'
        if idx == 7:
            padding = 'valid'
        x = _conv_bn_relu(filters=filter,
                          kernel_size=(3,3),
                          padding=padding,
                          strides=(1, 1),
                          need_activation=True,
                          conv_name="conv2d_%d" % idx,
                          bn_name="conv2d_%d_bn" % idx,
                          relu_name="conv2d_%d_activation" % idx)(x)
        if add_pool:
            x = MaxPool2D(pool_size=(2, 2), padding='same', name='maxpool2d_%d' % idx)(x)

    model = Model(img_input, x)
    return model
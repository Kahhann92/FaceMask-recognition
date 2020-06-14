# -*- encoding=utf-8 -*-
from keras.layers import Input, Conv2D, MaxPool2D, Activation, Reshape
from keras.layers import BatchNormalization
from keras.layers import ZeroPadding2D
from keras.regularizers import l2

def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu residual unit activation function.
       This is the original ResNet v1 scheme in https://arxiv.org/abs/1512.03385
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    need_activation = conv_params.setdefault("need_activation", False)

    def f(x):
        # 这个是在input_feature_map大小是偶数，步进为2的时候，为了与caffe的padding方式保持一致（caffe只能设置前后补同样位数的零）
        # if strides[0] == 2:
        #     x = ZeroPadding2D(((1, 0), (1, 0)), name=conv_name+"_pad")(x)
        #     padding_mode = 'valid'
        # else:
        #     padding_mode = padding
        x = Conv2D(filters=filters, kernel_size=kernel_size, use_bias=False,
                   strides=strides, padding=padding,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        if need_activation:
            x = Activation("relu", name=relu_name)(x)
            # x = PReLU(name=relu_name, shared_axes=[1,2])(x)
        return x
    return f


def cls_predictor(x,
                  num_anchors,
                  name,
                  num_classes = 2,
                  insert_addition_conv = True,
                  insert_addition_conv_filter = 64):
    if x.shape[1] < 3 or x.shape[2] < 3:
        kernel_size = (1, 1)
        print("This detection layer feature map is small than 3, use (1, 1) kernel size.")
    else:
        kernel_size = (3, 3)
    if insert_addition_conv:
        x = _conv_bn_relu(filters=insert_addition_conv_filter,
                          kernel_size=kernel_size,
                          padding='same',
                          strides=(1, 1),
                          need_activation=True,
                          conv_name=name + "_insert_conv2d",
                          bn_name=name + "_insert_conv2d_bn",
                          relu_name=name + "_insert_conv2d_activation")(x)
    x = Conv2D(filters= num_anchors * num_classes,
                  kernel_size=kernel_size,
                  padding="SAME", name=name + "_conv")(x)
    x = Reshape((-1, num_classes), name=name + "_reshape")(x)
    return Activation('sigmoid', name=name + "_activation")(x)


def bbox_predictor(x,
                   num_anchors,
                   name,
                   output_nodes = 8,
                   insert_addition_conv=True,
                   insert_addition_conv_filter=64,
                   ):
    if x.shape[1] < 3 or x.shape[2] < 3:
        kernel_size = (1, 1)
        print("This detection layer feature map is small than 3, use (1, 1) kernel size.")
    else:
        kernel_size = (3, 3)

    if insert_addition_conv:
        x = _conv_bn_relu(filters=insert_addition_conv_filter,
                          kernel_size=kernel_size,
                          padding='same',
                          strides=(1, 1),
                          need_activation=True,
                          conv_name=name + "_conv2d_1x1",
                          bn_name=name + "_conv2d_1x1_bn",
                          relu_name=name + "_conv2d_1x1_activation")(x)

    x = Conv2D(filters= num_anchors * output_nodes, kernel_size=kernel_size, padding="SAME", name=name + "_conv")(x)
    return Reshape((-1, output_nodes), name=name + "_reshape")(x)


def detect_head(x,
               num_anchors,
               name,
               node_per_anchor = 4,
               activation_type = None,
               insert_addition_conv_num=1,
               insert_addition_conv_filter=64):
    if x.shape[1] < 3 or x.shape[2] < 3:
        kernel_size = (1, 1)
        print("This detection layer feature map is small than 3, use (1, 1) kernel size.")
    else:
        kernel_size = (3, 3)

    for _ in range(insert_addition_conv_num):
        x = _conv_bn_relu(filters=insert_addition_conv_filter,
                          kernel_size=kernel_size,
                          padding='same',
                          strides=(1, 1),
                          need_activation=True,
                          conv_name=name + "_insert_conv2d",
                          bn_name=name + "_insert_conv2d_bn",
                          relu_name=name + "_insert_conv2d_activation")(x)
    x = Conv2D(filters= num_anchors * node_per_anchor,
                  kernel_size=kernel_size,
                  padding="SAME", name=name + "_conv")(x)
    x = Reshape((-1, node_per_anchor), name=name + "_reshape")(x)
    if activation_type not in ('sigmoid','softmax', None):
        raise ValueError("Activation function must be in ('sigmoid','softmax', None) ")
    if activation_type is None:
        return x
    elif activation_type == 'sigmoid':
        return Activation('sigmoid', name=name + "_activation")(x)
    elif activation_type == 'softmax':
        return Activation('softmax', name=name + "_activation")(x)
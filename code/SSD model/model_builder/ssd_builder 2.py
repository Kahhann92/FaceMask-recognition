# -*- coding:utf-8 -*-
from keras.layers import Concatenate
from keras.models import Model

from model_builder.model_builder_util import detect_head

def build_ssd_model(backbone_model_builder,
                    input_shape,
                    detection_layer_config,
                    num_class):
    '''

    :param backbone_model_builder: 构建backbone模型的函数，接受input_shape作为参数，返回backbone网络
    :param input_shape: 检测网络的输入大小，[height, width, channel]
    :param detection_layer_config: 检测头的配置信息
    :param num_class: 网络需要检测的类别数目
    :return:
    '''
    assert len(input_shape) == 3, "The input_shape must be [height, width, channel]"
    backbone_model = backbone_model_builder(input_shape)
    loc_branch = []
    cls_branch = []

    for idx, plugin_layer_name in enumerate(sorted(detection_layer_config.keys())):
        try:
            plugin_layer = backbone_model.get_layer(plugin_layer_name)
        except ValueError as e:
            print(e)

        num_anchor = len(detection_layer_config[plugin_layer_name]['aspect_ratio']) +\
                     len(detection_layer_config[plugin_layer_name]['scale']) -1

        insert_addition_conv_num = detection_layer_config[plugin_layer_name].get('insert_addition_conv_num',0)

        insert_addition_conv_filter = detection_layer_config[plugin_layer_name].get('insert_addition_conv_filter',64)

        loc = detect_head(plugin_layer.output,
                           num_anchor,
                           "loc_%d" % idx,
                           node_per_anchor = 4,
                           activation_type = None,
                           insert_addition_conv_num=insert_addition_conv_num,
                           insert_addition_conv_filter=insert_addition_conv_filter)

        cls = detect_head(plugin_layer.output,
                           num_anchor,
                           "cls_%d" % idx,
                           node_per_anchor = num_class,
                           activation_type = 'sigmoid',
                           insert_addition_conv_num=insert_addition_conv_num,
                           insert_addition_conv_filter=insert_addition_conv_filter)

        loc_branch.append(loc)
        cls_branch.append(cls)

    loc_merged = Concatenate(axis=1, name='loc_branch_concat')(loc_branch)
    cls_merged = Concatenate(axis=1, name='cls_branch_concat')(cls_branch)

    ssd_model = Model(backbone_model.input, outputs=[loc_merged, cls_merged])
    return ssd_model




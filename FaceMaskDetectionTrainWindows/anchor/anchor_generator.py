# -*- encoding=utf-8 -*-
import numpy as np

def generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios, offset=0.5):
    '''
    :param feature_map_sizes: list of list, for example: [[40,40], [20,20]]
    :param anchor_sizes: list of list, for example: [[0.05, 0.075], [0.1, 0.15]]
    :param anchor_ratios: list of list, for example: [[1, 0.5], [1, 0.5]]
    :param offset: default to 0.5
    :return:
    '''
    anchor_bboxes = []
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
        cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
        cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
        center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

        num_anchors = len(anchor_sizes[idx]) +  len(anchor_ratios[idx]) - 1
        center_tiled = np.tile(center, (1, 1, 2* num_anchors))
        anchor_width_heights = []

        # different scales with the first same aspect ratio (1)
        # 选择第一个ratio，然后所有尺度使用这个ratio来生成不同尺度大小的anchor
        # 例如大小为[0.05, 0.075, 0.1]的scale分别用ratio=1生成anchor
        for scale in anchor_sizes[idx]:
            ratio = anchor_ratios[idx][0] # 选择第一个ratio
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        # the first scale, with different aspect ratios (except the first one)
        # 第一个尺度，与除了第一个以外的剩余ratio计算，例如[0.5, 2, 0.333, 5]
        for ratio in anchor_ratios[idx][1:]:
            s1 = anchor_sizes[idx][0] # 选择第一个scale
            width = s1 * np.sqrt(ratio)
            height = s1 / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        bbox_coords = center_tiled + np.array(anchor_width_heights)
        bbox_coords_reshape = bbox_coords.reshape((-1, 4))
        anchor_bboxes.append(bbox_coords_reshape)
    anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
    return anchor_bboxes


if __name__ == '__main__':
    feature_map_sizes = [[88, 50], [44, 25], [22, 13], [11, 7], [6, 4], [3, 2]]
    anchor_sizes = [[0.03, 0.05], [0.06, 0.08], [0.12, 0.20], [0.24, 0.40], [0.48, 0.6], [0.8, 0.9]]
    anchor_ratios = [[0.56], [0.56], [0.56, 0.25], [0.56, 0.25], [0.56, 0.25], [0.56, 0.25]]
    anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
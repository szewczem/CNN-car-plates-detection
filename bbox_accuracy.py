import numpy as np
import tensorflow as tf


def intersection_over_union(rescaled_bboxs_predicted, rescaled_bbox_original):
    '''
    IoU - Intersection over Union, measure the overlap between two bounding boxes.

    Arguments:
        bbox_predicted -- resized predicted bbox coordinates
        bbox_original  -- resized original bbox coordinates
    Return:
        iou -- the accuracy of predicted bounding boxes against ground truth bounding boxes, 0 - poor accuracy, 1 - perfect accuracy
    '''

    pred = np.array(rescaled_bboxs_predicted)
    true = np.array(rescaled_bbox_original)

    # Find cross-overs coordinates, (xA, yA) - top left point of intersection, (xB, yB) - bottom right point of intersection
    xA = np.amax([pred[:, 0], true[:, 0]], axis=0)
    yA = np.amax([pred[:, 1], true[:, 1]], axis=0)
    xB = np.amin([pred[:, 2], true[:, 2]], axis=0)
    yB = np.amin([pred[:, 3], true[:, 3]], axis=0)

    # Intersection width and height
    # if xB < xA or yB < yA, then boxes do not overlap, its width or height is then equal to 0
    inter_width = np.maximum(0, xB - xA)
    inter_height = np.maximum(0, yB - yA)

    # Area of intersection
    inter_area = inter_width * inter_height
    # end if intersection doesn't exis
    # if inter_area == 0:
    #     iou = 0.0
    #     continue

    # Area of predicted bbox and original bbox
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    true_area = (true[:, 2] - true[:, 0]) * (true[:, 3] - true[:, 1])

    # IoU = intersection_area / union_area   =>   IoU = intersection_area / (predicted_area + original_area - intersection_area)
    iou = inter_area / (pred_area + true_area - inter_area)

    return iou.tolist()


'''
For Tensorflow model
'''
def mean_iou(y_true, y_pred):
    # x_min, y_min, x_max, y_max for both
    x1 = tf.maximum(y_true[:, 0], y_pred[:, 0])
    y1 = tf.maximum(y_true[:, 1], y_pred[:, 1])
    x2 = tf.minimum(y_true[:, 2], y_pred[:, 2])
    y2 = tf.minimum(y_true[:, 3], y_pred[:, 3])

    intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)

    area_true = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    area_pred = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])

    union = area_true + area_pred - intersection
    iou = tf.math.divide_no_nan(intersection, union)

    return tf.reduce_mean(iou)
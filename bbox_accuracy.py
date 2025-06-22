import numpy as np
import tensorflow as tf
from data_preparation import rescale_bbox


def intersection_over_union(rescaled_bbox_original, rescaled_bboxs_predicted):
    '''
    IoU - Intersection over Union, measure the overlap between two bounding boxes.

    Arguments:
        bbox_predicted -- resized predicted bbox coordinates
        bbox_original  -- resized original bbox coordinates
    Return:
        iou -- the accuracy of predicted bounding boxes against ground truth bounding boxes, 0 - poor accuracy, 1 - perfect accuracy
    '''

    true = np.array(rescaled_bbox_original)
    pred = np.array(rescaled_bboxs_predicted)

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

    # Area of predicted bbox and true (original) bbox
    true_area = (true[:, 2] - true[:, 0]) * (true[:, 3] - true[:, 1])
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])

    # IoU = intersection_area / union_area   =>   IoU = intersection_area / (predicted_area + original_area - intersection_area)
    iou = inter_area / (pred_area + true_area - inter_area)

    return iou.tolist()


'''
For Keras model, the IoU is counted for not rescaled bboxes
'''
def mean_iou(y_true, y_pred):
    # Cross-overs coordinates, (xA, yA) - top left point of intersection, (xB, yB) - bottom right point of intersection
    xA = tf.maximum(y_true[:, 0], y_pred[:, 0])
    yA = tf.maximum(y_true[:, 1], y_pred[:, 1])
    xB = tf.minimum(y_true[:, 2], y_pred[:, 2])
    yB = tf.minimum(y_true[:, 3], y_pred[:, 3])

    # Area of intersection
    inter_area = tf.maximum(0.0, xB - xA) * tf.maximum(0.0, yB - yA)

    # Area of predicted bbox and true (original) bbox
    true_area = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    pred_area = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])

    # IoU = intersection_area / (predicted_area + original_area - intersection_area)
    iou = inter_area / (true_area + pred_area - inter_area)

    return tf.reduce_mean(iou)
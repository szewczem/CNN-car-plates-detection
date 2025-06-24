import numpy as np
import tensorflow as tf
from data_preparation import rescale_bbox
from keras.saving import register_keras_serializable


def mean_iou(rescaled_bbox_original, rescaled_bboxs_predicted):
    '''
    IoU - Intersection over Union, measure the overlap between two bounding boxes.

    Arguments:
        bbox_predicted -- resized predicted bbox coordinates
        bbox_original -- resized original bbox coordinates
    Return:
        iou -- the accuracy of predicted bounding boxes against ground truth bounding boxes, 0 - poor accuracy, 1 - perfect accuracy
    '''

    true = np.array(rescaled_bbox_original)
    pred = np.array(rescaled_bboxs_predicted)

    # Top-left values are always lower than bottom-right, the (0,0) point is on the top left corner
    xtl_true = np.minimum(true[:, 0], true[:, 2])
    xbr_true = np.maximum(true[:, 0], true[:, 2])
    ytl_true = np.minimum(true[:, 1], true[:, 3])
    ybr_true = np.maximum(true[:, 1], true[:, 3])
    
    xtl_pred = np.minimum(pred[:, 0], pred[:, 2])
    xbr_pred = np.maximum(pred[:, 0], pred[:, 2])
    ytl_pred = np.minimum(pred[:, 1], pred[:, 3])
    ybr_pred = np.maximum(pred[:, 1], pred[:, 3])

    # Find cross-overs coordinates, (xA, yA) - top left point of intersection, (xB, yB) - bottom right point of intersection
    xA = np.maximum(xtl_true, xtl_pred)
    yA = np.maximum(ytl_true, ytl_pred)
    xB = np.minimum(xbr_true, xbr_pred)
    yB = np.minimum(ybr_true, ybr_pred)

    # Intersection width and height
    # if xB < xA or yB < yA, then boxes do not overlap, its width or height is then equal to 0
    inter_width = np.maximum(0, xB - xA)
    inter_height = np.maximum(0, yB - yA)

    # Area of intersection
    inter_area = inter_width * inter_height

    # Area of predicted bbox and true (original) bbox, 1e-5 for valid output if true_area is 0 (for image with no bbox, not in current data set)
    true_area = (xbr_true - xtl_true) * (ybr_true - ytl_true) + 1e-5
    pred_area = (xbr_pred - xtl_pred) * (ybr_pred - ytl_pred) + 1e-5

    # IoU = intersection_area / union_area   =>   IoU = intersection_area / (predicted_area + original_area - intersection_area)
    iou = inter_area / (pred_area + true_area - inter_area)    

    # Return the mean value
    return np.mean(iou)


@register_keras_serializable()
def mean_iou_keras(y_true, y_pred):
    '''
    For Keras model, the IoU is counted for not rescaled bboxes

    Arguments:
            y_true -- original bbox coordinates (normalized)
            y_pred -- predicted bbox coordinates (normalized)
        Return:
            iou -- the accuracy of predicted bounding boxes against ground truth bounding boxes, 0 - poor accuracy, 1 - perfect accuracy
    '''

    # Top-left values are always lower than bottom-right, the (0,0) point is on the top left corner
    xtl_true = tf.minimum(y_true[:, 0], y_true[:, 2])
    xbr_true = tf.maximum(y_true[:, 0], y_true[:, 2])
    ytl_true = tf.minimum(y_true[:, 1], y_true[:, 3])
    ybr_true = tf.maximum(y_true[:, 1], y_true[:, 3])
    
    xtl_pred = tf.minimum(y_pred[:, 0], y_pred[:, 2])
    xbr_pred = tf.maximum(y_pred[:, 0], y_pred[:, 2])
    ytl_pred = tf.minimum(y_pred[:, 1], y_pred[:, 3])
    ybr_pred = tf.maximum(y_pred[:, 1], y_pred[:, 3])

    # Cross-overs coordinates, (xA, yA) - top left point of intersection, (xB, yB) - bottom right point of intersection
    xA = tf.maximum(xtl_true, xtl_pred)
    yA = tf.maximum(ytl_true, ytl_pred)
    xB = tf.minimum(xbr_true, xbr_pred)
    yB = tf.minimum(ybr_true, ybr_pred)

    # Area of intersection
    inter_area = tf.maximum(0.0, xB - xA) * tf.maximum(0.0, yB - yA)

    # Area of predicted bbox and true (original) bbox, 1e-5 for valid output if true_area is 0 (for image with no bbox, not in current data set)
    true_area = (xbr_true - xtl_true) * (ybr_true - ytl_true) + 1e-5
    pred_area = (xbr_pred - xtl_pred) * (ybr_pred - ytl_pred) + 1e-5

    # IoU = intersection_area / (predicted_area + original_area - intersection_area)
    iou = inter_area / (true_area + pred_area - inter_area)

    # Return the mean value
    return tf.reduce_mean(iou)
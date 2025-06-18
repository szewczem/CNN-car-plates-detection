import numpy as np


def intersection_over_union(rescaled_bboxs_predicted, rescaled_bbox_original):
    '''
    IoU - Intersection over Union, measure the overlap between two bounding boxes.

    Arguments:
        bbox_predicted -- resized predicted bbox coordinates
        bbox_original  -- resized original bbox coordinates
    Return:
        iou -- the accuracy of predicted bounding boxes against ground truth bounding boxes, 0 - poor accuracy, 1 - perfect accuracy
    '''
    iou_list = []
    for bbox_predicted, bbox_original in zip(rescaled_bboxs_predicted, rescaled_bbox_original):
        xtl_pred, ytl_pred, xbr_pred, ybr_pred = bbox_predicted
        xtl_ori, ytl_ori, xbr_ori, ybr_ori = bbox_original

        # Find cross-overs coordinates
        xA = max(xtl_pred, xtl_ori)
        yA = max(ytl_pred, ytl_ori)
        xB = min(xbr_pred, xbr_ori)
        yB = min(ybr_pred, ybr_ori)

        # Intersection width and height
        # if xB < xA or yB < yA, then boxes do not overlap, its width or height is then equal to 0
        inter_width = max(0, xB - xA)
        inter_height = max(0, yB - yA)

        # Area of intersection
        inter_area = inter_width * inter_height
        # end if intersection doesn't exis
        if inter_area == 0:
            iou_list.append(0.0)
            continue

        # Area of predicted bbox and original bbox
        predicted_area = (xbr_pred - xtl_pred) * (ybr_pred - ytl_pred)
        original_area = (xbr_ori - xtl_ori) * (ybr_ori - ytl_ori)

        # IoU = intersection_area / union_area   =>   IoU = intersection_area / (predicted_area + original_area - intersection_area)
        iou = inter_area / (predicted_area + original_area - inter_area)

        iou_list.append(float(iou))
    return iou_list
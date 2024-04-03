# this file convert yolo number format to digital format
from icecream import ic
import numpy as np


def convert_number_to_frame(img_dim, bb):
    if (bb is None) or (len(bb) == 0):
        return []

    img_w, img_h = img_dim[0], img_dim[1]
    bb = bb.copy()

    # convert bb from roboflow format to xyxy format
    for index, _bb in enumerate(bb):
        cls = _bb[0]
        _bb_x, _bb_y, _bb_w, _bb_h = _bb[1:]

        x_min, x_max = _bb_x - (_bb_w / 2), _bb_x + (_bb_w / 2)
        y_min, y_max = _bb_y - (_bb_h / 2), _bb_y + (_bb_h / 2)
        xyxy_bb = np.array([cls, x_min * img_w, y_min * img_h, x_max * img_w, y_max * img_h]) 
        bb[index] = xyxy_bb
    
    # find the frame bb
    x_min, y_min, x_max, y_max = bb[0][1:]
    for index, _bb in enumerate(bb):
        # check x min 
        if x_min > _bb[1]:
            x_min = _bb[1]
        
        if y_min > _bb[2]:
            y_min = _bb[2]
        
        if x_max < _bb[3]:
            x_max = _bb[3]
        
        if y_max < _bb[4]:
            y_max = _bb[4]
    
    # use this for visualize
    # x_min = max(0, x_min)
    # y_min = max(0, y_min)
    # x_max = min(x_max, img_w-1)
    # y_max = min(y_max, img_h-1)

    # use this for real
    x_min = max(0, x_min) / img_w
    y_min = max(0, y_min) / img_h
    x_max = min(x_max, img_w-1) / img_w
    y_max = min(y_max, img_h-1) / img_h
        
    return np.array([[2,x_min, y_min, x_max, y_max]])

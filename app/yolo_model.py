import copy
import io

import cv2
import numpy as np
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords

from . import utils

# Model Parameters
DEVICE = 'cpu'
CONF_THRESH = 0.4  # Object confidence threshold
IOU_THRESH = 0.45  # IOU threshold for NMS
CLASS_MAP = []


class YOLOv5Model(object):
    
    def __init__(self, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH, class_map=CLASS_MAP, device=DEVICE):
        self._model = None
        self._stride = None
        self._imgsz = None
        self._conf_thresh = None
        self._iou_thresh = None
        self._nms_class = None
        self._agnostic_nms = False
        self._class_map = []
        self._device = device
        
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.class_map = class_map

    @property
    def model(self):
        return self._model
    
    @property
    def stride(self):
        return self._stride
    
    @property
    def imgsz(self):
        return self._imgsz
    
    @imgsz.setter
    def imgsz(self, value):
        assert self._stride, 'Model must be loaded before imgsz can be set.'
        self._imgsz = check_img_size(value, s=self._stride)
        
    @property
    def conf_thresh(self):
        return self._conf_thresh
    
    @conf_thresh.setter
    def conf_thresh(self, value):
        assert value >= 0 and value <= 1, 'Confidence threshold must be between 0 and 1.'
        self._conf_thresh = value
        
    @property
    def iou_thresh(self):
        return self._iou_thresh
    
    @iou_thresh.setter
    def iou_thresh(self, value):
        assert value >= 0 and value <= 1, 'IOU threshold must be between 0 and 1.'
        self._iou_thresh = value
        
    @property
    def class_map(self):
        return copy.deepcopy(self._class_map)
    
    @class_map.setter
    def class_map(self, value):
        assert type(value) == list and len(value) > 0, 'Class map must be a list of length > 0'
        self._class_map = value

    def __call__(self, img_data):
        assert self._model, 'Model must be loaded before inferencing.'
        assert self._imgsz, 'Image size must be set before inferencing.'
        img, imgsz0 = self.preprocess(img_data)
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, 
                                   self._nms_class, agnostic=self._agnostic_nms)[0]
        pred = self.postprocess(pred, imgsz0, img_data.shape[:2])
        return pred
    
    def preprocess(self, img):
        """ Prepare the input for inferencing. """
        imgsz0 = torch.Tensor(img.shape[:2])
        
        # resize image
        img = letterbox(img, self.imgsz, stride=self.stride)[0]

        # convert from BGR to RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # convert to tensor
        img = torch.from_numpy(img).to(self._device)

        # normalize RGB values to percentage
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img, imgsz0
    
    def postprocess(self, predictions, imgsz0, imgsz1):
        """ Convert class IDs to class names. """
        predictions[:, :4] = scale_coords(imgsz1, predictions[:, :4], imgsz0).round()
        predictions = predictions.cpu().numpy().tolist()
        return [{"box": row[:4],
                 "confidence": row[4],
                 "class": self._class_map[int(row[5])]} for row in predictions]

    def from_s3(self, s3_client, bucket, key):
        """ Load the YoloV5 model. """
        s3_client.download_file(bucket, key, '/tmp/weights.pt')
        self.from_file('/tmp/weights.pt')

    def from_file(self, file):
        self._model = attempt_load(file, map_location=self._device)
        self._stride = int(self._model.stride.max())
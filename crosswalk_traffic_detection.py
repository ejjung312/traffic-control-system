# import sys
# sys.path.insert(0, './model')

import cv2
import torch
import numpy as np

from ultralytics import YOLO
# from utils.datasets import letterbox
from ultralytics.data.dataset import LetterBox
# from utils.general import non_max_suppression, scale_coords
from ultralytics.utils import ops
from ultralytics.utils.plotting import Annotator

MODEL_PATH = 'model/crosswalk_best.pt'

img_size = 640
conf_threshold = 0.5
iou_threshold = 0.45
max_detection = 1000
classes = None
agnostic_nms = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ckpt = torch.load(MODEL_PATH, map_location=device)
# model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()
model = YOLO(MODEL_PATH)
class_names = ['횡단보도', '빨간불', '초록불'] # model.names
stride = int(model.stride.max())
colors = ((50,50,50), (0,0,255), (0,255,0)) # (gray, red, green)

cap = cv2.VideoCapture('datasets/sample.mp4')

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break


    # img_input = letterbox(img, img_size, stride=stride)[0]
    letterbox = LetterBox(new_shape=(640, 640), stride=stride)
    img_input = letterbox(img)
    # transpose((2,0,1)) ===> (H,W,C) -> (C,H,W)로 변경. 파이토치는 (C,H,W)형태여야 함
    # [::-1] ===> [R,G,B] -> [B,G,R]로 변경. OpenCV 형식에 맞추기 위해
    img_input = img_input.transpose((2,0,1))[::-1]
    img_input = np.ascontiguousarray(img_input)
    img_input = torch.from_numpy(img_input).to(device)
    img_input = img_input.float()
    img_input /= 255.
    img_input = img_input.unsqueeze(0)

    # 인퍼런스
    pred = model(img_input, augment=False, visualize=False)[0]

    # postprocess
    # pred = ops.non_max_suppression(pred, conf_threshold, iou_threshold, classes, agnostic_nms, max_det=max_detection)[0]
    pred = ops.non_max_suppression(pred, conf_threshold, iou_threshold, classes, agnostic_nms, max_det=max_detection)

    pred = pred.cpu().numpy()

    # pred[:,:4] = ops.scale_coords(img_input.shape[2:], pred[:,:4], img.shape).round()
    ops.scale_coords(img_input.shape[2:], pred[:,:4], img.shape)

    annotator = Annotator(img.copy(), line_width=3, example=str(class_names))

    for p in pred:
        class_name = class_names[int(p[5])]

        x1,y1,x2,y2 = p[:4]

        annotator.box_label([x1,y1,x2,y2], '%s %d' % (class_name, float(p[4]) * 100), color=colors[int(p[5])])

    result_img = annotator.result()

    cv2.imshow('result', result_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()




















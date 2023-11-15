# -*- coding: UTF-8 -*-

from pathlib import Path

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy

# from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized



def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def show_results(img, xyxy, conf, landmarks, class_num):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1, y1, x2, y2 = map(int, xyxy)
    img = img.copy()

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)
        cv2.putText(img, f"({point_x}, {point_y})", (point_x, point_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [225, 255, 255], thickness=tl, lineType=cv2.LINE_AA)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    # Display xyxy coordinates
    text_margin = 5
    text_height = cv2.getTextSize("Text", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][1]
    text_y = y1 + text_height + text_margin
    cv2.putText(img, f"x1: {x1}", (x1 + text_margin, y1 + text_height + text_margin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [225, 255, 255], thickness=tl, lineType=cv2.LINE_AA)
    cv2.putText(img, f"y1: {y1}", (x1 + text_margin, y1 + 2 * (text_height + text_margin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [225, 255, 255], thickness=tl, lineType=cv2.LINE_AA)
    cv2.putText(img, f"x2: {x2}", (x2 - text_margin, y2 - 2 * (text_height + text_margin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [225, 255, 255], thickness=tl, lineType=cv2.LINE_AA)
    cv2.putText(img, f"y2: {y2}", (x2 - text_margin, y2 - (text_height + text_margin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [225, 255, 255], thickness=tl, lineType=cv2.LINE_AA)

    return img

import cv2
import numpy as np
import copy
from pathlib import Path
import torch

def detect(
    model,
    source,
    device,
    project,
    name,
    exist_ok,
    save_img,
    view_img
):
    # Load model
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5
    imgsz = (640, 640)
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    is_file = Path(source).suffix[1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    landmarks_dict = {}
    
    # Dataloader
    if webcam:
        print('loading streams:', source)
        dataset = LoadStreams(source, img_size=imgsz)
        bs = 1  # batch_size
    else:
        print('loading images', source)
        dataset = LoadImages(source, img_size=imgsz)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    exit_flag = False  # Flag to check if 'Q' key is pressed
    frame_counter = 0
    for path, im, im0s, vid_cap in dataset:
        if exit_flag:
            break

        if len(im.shape) == 4:
            orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis=0)
        else:
            orgimg = im.transpose(1, 2, 0)

        orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            xyxy = [0,0,0,0]
            landmarks = [0,0,0,0,0,0,0,0,0,0]
            if exit_flag:
                break

            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(Path(save_dir) / p.name)  # im.jpg

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()

                max_area = 0
                max_det = None

                # Calculate area and select detection with max area
                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])  # Calculate area
                    if area > max_area:
                        max_area = area
                        max_det = det[j]

                # If a face was detected, draw it on the image
                if max_det is not None:
                    xyxy = max_det[:4].view(-1).tolist()
                    conf = max_det[4].cpu().numpy()
                    landmarks = max_det[5:15].view(-1).tolist()
                    class_num = max_det[15].cpu().numpy()

                    im0 = show_results(im0, xyxy, conf, landmarks, class_num)

            if view_img:
                cv2.imshow('result', im0)
                k = cv2.waitKey(1)
                if k == ord('q') or k == ord('Q'):  # Check if 'Q' key is pressed
                    exit_flag = True

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    try:
                        vid_writer[i].write(im0)
                        result = xyxy + landmarks
                        landmarks_dict[frame_counter] = result #将face_landmark存在字典
                        frame_counter+=1
                    except Exception as e:
                        print(e)
    
    if view_img:
        cv2.destroyAllWindows()
                    
            


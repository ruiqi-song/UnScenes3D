#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2024-03-26 17:39:57
Description: 
LastEditors: knightdby
LastEditTime: 2025-05-16 14:18:00
FilePath: /UnScenes3D/pipline/uns_is/app.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-05-15 17:11:00
"""

from manifast import *

try:
    import detectron2
except:
    import os
    os.system('pip install git+https://github.com/facebookresearch/detectron2.git')
    # os.system('git clone https://github.com/facebookresearch/detectron2.git')
    # os.system('python -m pip install -e detectron2')

import cv2
import math
import torch
import torchvision
import gradio as gr
import numpy as np
import torch.nn.functional as F
from detectron2.config import get_cfg
from pipline.uns_is.glee.models.glee_model import GLEE_Model
from pipline.uns_is.glee.config import add_glee_config

print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(
        f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def scribble2box(img):
    if img.max() == 0:
        return None, None
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    all = np.any(img, axis=2)
    R, G, B, A = img[np.where(all)[0][0], np.where(all)[
        1][0]].tolist()  # get color
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return np.array([xmin, ymin, xmax, ymax]), (R, G, B)


def getBinary(img_or_path, minConnectedArea=20):
    if isinstance(img_or_path, str):
        i = cv2.imread(img_or_path)
    elif isinstance(img_or_path, np.ndarray):
        i = img_or_path
    else:
        raise TypeError('Input type error')

    if len(i.shape) == 3:
        img_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

    else:
        img_gray = i
    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin)
    for index in range(1, stats.shape[0]):
        if stats[index][4] < minConnectedArea or stats[index][4] < 0.1 * (
                stats[index][2] * stats[index][3]):
            labels[labels == index] = 0
    labels[labels != 0] = 1
    img_bin = np.array(img_bin * labels).astype(np.uint8)
    return i, img_bin


def get_approx(img, contour, length_p=0.1):
    """获取逼近多边形

    :param img: 处理图片
    :param contour: 连通域
    :param length_p: 逼近长度百分比
    """
    img_adp = img.copy()
    # 逼近长度计算
    epsilon = length_p * cv2.arcLength(contour, True)
    # 获取逼近多边形
    approx = cv2.approxPolyDP(contour, epsilon, True)

    return approx


currentCV_version = cv2.__version__  # str


def getMultiRegion(img, img_bin):
    """
    for multiple objs in same class
    """
    # tmp = currentCV_version.split('.')
    if float(currentCV_version[0:3]) < 3.5:
        img_bin, contours, hierarchy = cv2.findContours(
            img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    if len(contours) >= 1:

        for i in range(0, len(contours)):
            area = cv2.contourArea(contours[i])
            region = get_approx(img, contours[i], 0.002)
            if region.shape[0] > 3 and area > 20:
                regions.append(region)
        return regions
    else:
        return []


def getPolygonwithMask(oriImg):
    img, img_bin = getBinary(oriImg)
    return getMultiRegion(img, img_bin)


def LSJ_box_postprocess(out_bbox,  padding_size, crop_size, img_h, img_w):
    boxes = box_cxcywh_to_xyxy(out_bbox)
    lsj_sclae = torch.tensor(
        [padding_size[1], padding_size[0], padding_size[1], padding_size[0]]).to(out_bbox)
    crop_scale = torch.tensor(
        [crop_size[1], crop_size[0], crop_size[1], crop_size[0]]).to(out_bbox)
    boxes = boxes * lsj_sclae
    boxes = boxes / crop_scale
    boxes = torch.clamp(boxes, 0, 1)

    scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
    scale_fct = scale_fct.to(out_bbox)
    boxes = boxes * scale_fct
    return boxes


class NearNeighborRemover:
    def __init__(self, distance_threshold):
        self.distance_threshold = distance_threshold

    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def remove_near_neighbors(self, points):
        # Add the first point to the filtered list
        filtered_points = [points[0]]
        for i in range(1, len(points)):
            # Calculate the distance between the current point and the last added point
            distance = self.calculate_distance(points[i], filtered_points[-1])
            # If the distance is above the threshold, add the current point to the filtered list
            if distance >= self.distance_threshold:
                filtered_points.append(points[i])
        return filtered_points


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
          [0.494, 0.000, 0.556], [0.494, 0.184, 0.741], [0.301, 0.745, 0.000],
          [0.700, 0.300, 0.600], [0.000, 0.447, 0.741], [0.850, 0.325, 0.098]]


coco_class_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
OBJ365_class_names = ['object']
class_agnostic_name = ['object']

if torch.cuda.is_available():
    print('use cuda')
    device = 'cuda'
else:
    print('use cpu')
    device = 'cpu'
# TODO:
cfg_r50 = get_cfg()
add_glee_config(cfg_r50)

weigths = [
    'weights/unscene_2d/uns_is_model_weights.pth',
]
conf_files_r50 = osp.dirname(weigths[0])+'/config.yaml'
cfg_r50.merge_from_file(conf_files_r50)
cfg_r50.OUTPUT_DIR = './work_dirs/val/'+cfg_r50.OUTPUT_DIR
cfg_r50.MODEL.WEIGHTS = weigths[0]


cfg_r50.SOLVER.IMS_PER_BATCH = 1
cfg_r50.freeze()


GLEEmodel_rawglee_r50 = None

model = GLEE_Model(cfg_r50, None, device, None, True).to(device)
checkpoints_r50 = torch.load(cfg_r50.MODEL.WEIGHTS)
model.load_state_dict(checkpoints_r50, strict=False)
model.eval()


pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).to(device).view(3, 1, 1)
pixel_std = torch.Tensor([58.395, 57.12, 57.375]).to(device).view(3, 1, 1)
def normalizer(x): return (x - pixel_mean) / pixel_std


inference_size = 800
inference_type = 'resize_shot'  # or LSJ
size_divisibility = 32

FONT_SCALE = 1e-3
THICKNESS_SCALE = 1e-3
TEXT_Y_OFFSET_SCALE = 1e-2
cate_color_map = {
    'Retainingwall': (128, 0, 0),
    'Road': (0, 128, 0),
    'Rock': (255, 128, 0),
    'Warningsign': (255, 0, 0),
    'Signboard': (255, 0, 0),
    'Puddle': (0, 0, 128),
    'Slit': (128, 128, 128),
    'Rut': (128, 128, 0),
    'Cable': (128, 64, 255)}

if inference_type != 'LSJ':
    resizer = torchvision.transforms.Resize(inference_size)


def get_examples(sence_name, cate_name='object'):
    files = get_files('assets/examples', in_line=sence_name)
    paths = []
    for path in files:
        if cate_name not in path:
            continue
        paths.append(path)
    return paths


def segment_image(img, prompt_mode, categoryname, custom_category, expressiong, results_select, num_inst_select, base_threshold, instance_threshold, semiseg_threshold, mask_image_mix_ration, model_selection, export_result):
    GLEEmodel_list = [model, GLEEmodel_rawglee_r50]
    pred_class = None
    pred_boxes = None
    pred_mask = None
    pred_scores = None
    batch_category_list = []
    if prompt_mode == 'categories' and categoryname == "Custom-List":
        if ';' in custom_category:
            batch_category_list = [custom_category.split('; ')[0].split(
                ','), custom_category.split('; ')[1].split(',')]
        else:
            batch_category_list = [custom_category.split(',')]
    model_num = 2
    if GLEEmodel_rawglee_r50 is None:
        model_num = 1
    caption_pred = ''
    for idx in range(model_num):

        GLEEmodel = GLEEmodel_list[idx]
        copyed_img = img

        ori_image = torch.as_tensor(
            np.ascontiguousarray(copyed_img.transpose(2, 0, 1)))
        ori_image = normalizer(ori_image.to(device))[None,]
        _, _, ori_height, ori_width = ori_image.shape

        if inference_type == 'LSJ':
            infer_image = torch.zeros(1, 3, 1024, 1024).to(ori_image)
            infer_image[:, :, :inference_size, :inference_size] = ori_image
        else:
            resize_image = resizer(ori_image)
            image_size = torch.as_tensor(
                (resize_image.shape[-2], resize_image.shape[-1]))
            re_size = resize_image.shape[-2:]
            if size_divisibility > 1:
                stride = size_divisibility
                padding_size = ((image_size + (stride - 1)).div(stride,
                                rounding_mode="floor") * stride).tolist()
                infer_image = torch.zeros(
                    1, 3, padding_size[0], padding_size[1]).to(resize_image)
                infer_image[0, :, :image_size[0],
                            :image_size[1]] = resize_image

        if prompt_mode == 'categories' or prompt_mode == 'expression':
            if len(results_select) == 0:
                results_select = ['box']
            if prompt_mode == 'categories':
                if categoryname == "COCO-80":
                    batch_category_name = coco_class_name
                elif categoryname == "OBJ365":
                    batch_category_name = OBJ365_class_names
                elif categoryname == "Custom-List":
                    batch_category_name = batch_category_list[idx]
                else:
                    batch_category_name = class_agnostic_name

                prompt_list = []
                with torch.no_grad():
                    if idx == 0:
                        (outputs, _), _, _ = GLEEmodel(infer_image, prompt_list, task="coco",
                                                       batch_name_list=batch_category_name, is_train=False)
                    else:
                        (outputs, _), _, _ = GLEEmodel(infer_image, prompt_list, task="coco",
                                                       batch_name_list=batch_category_name, is_train=False)

                topK_instance = max(num_inst_select, 1)
            else:
                topK_instance = 1
                prompt_list = {'grounding': [expressiong]}
                with torch.no_grad():
                    (outputs, _), _, _, _ = GLEEmodel(infer_image, prompt_list,
                                                      task="grounding", batch_name_list=[], is_train=False)
            mask_pred = outputs['pred_masks'][0]
            mask_cls = outputs['pred_logits'][0]
            boxes_pred = outputs['pred_boxes'][0]
            if idx == 0:
                if 'pred_captions' in outputs:
                    caption_pred = outputs['pred_captions'][0]
                else:
                    caption_pred = ''

            scores = mask_cls.sigmoid().max(-1)[0]
            scores_per_image, topk_indices = scores.topk(
                topK_instance, sorted=True)
            if prompt_mode == 'categories':
                valid = scores_per_image > base_threshold
                topk_indices = topk_indices[valid]
                scores_per_image = scores_per_image[valid]
            pred_class_per = mask_cls[topk_indices].max(-1)[1].tolist()
            pred_boxes_per = boxes_pred[topk_indices]
            pred_mask_per = mask_pred[topk_indices]
            if pred_class is None:
                pred_class = [batch_category_name[cls]
                              for cls in pred_class_per]
            else:
                pred_class.extend([batch_category_name[cls]
                                   for cls in pred_class_per])
            if pred_scores is None:
                pred_scores = scores_per_image
            else:
                pred_scores = torch.cat([pred_scores, scores_per_image], dim=0)
            if pred_boxes is None:
                pred_boxes = pred_boxes_per
            else:
                pred_boxes = torch.cat([pred_boxes, pred_boxes_per], dim=0)
            if pred_mask is None:
                pred_mask = pred_mask_per
            else:
                pred_mask = torch.cat([pred_mask, pred_mask_per], dim=0)
    if pred_mask[None,].shape[1] == 0:
        return copyed_img, {}, caption_pred
    boxes = LSJ_box_postprocess(
        pred_boxes, padding_size, re_size, ori_height, ori_width)
    pred_masks = F.interpolate(pred_mask[None,], size=(
        padding_size[0], padding_size[1]), mode="bilinear", align_corners=False)
    pred_masks = pred_masks[:, :, :re_size[0], :re_size[1]]
    pred_masks = F.interpolate(pred_masks, size=(
        ori_height, ori_width), mode="bilinear", align_corners=False)
    pred_masks = (pred_masks > 0).detach().cpu().numpy()[0]

    export_result = {}

    if 'mask' in results_select:
        zero_mask = np.zeros_like(copyed_img)
        for nn, (label, mask) in enumerate(zip(pred_class, pred_masks)):
            label = label.replace(' ', '').capitalize()

            id_str = str(nn)
            export_result[id_str] = {}
            export_result[id_str]['label'] = ''
            export_result[id_str]['score'] = ''
            export_result[id_str]['area'] = 0.0
            export_result[id_str]['bboxes'] = []
            export_result[id_str]['masks'] = []

            score = pred_scores[nn].item()

            RGB = (COLORS[nn % 12][2]*255, COLORS[nn %
                   12][1]*255, COLORS[nn % 12][0]*255)
            if label in cate_color_map:
                RGB = cate_color_map[label][::-1]
            if (label == 'Retainingwall' or label == 'Road'):
                if score < semiseg_threshold:
                    continue
                if label == 'Retainingwall':
                    RGB = (0, 0, 128)
                else:
                    RGB = (0, 128, 0)

            width = mask.shape[1]
            height = mask.shape[0]
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
            seg_map_cls = mask.astype(np.uint8)
            contours, hierarchy = cv2.findContours(
                seg_map_cls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points_list = []
            points = []
            area = 0.0
            for contour in contours:
                pts = []
                for point in contour:
                    x, y = point[0]
                    pts.append([float(x)/width*100,
                                float(y)/height * 100])
                area_sub = cv2.contourArea(
                    np.array(pts).astype(np.int32))
                if area_sub > 60:
                    distance_threshold = 1.5
                else:
                    distance_threshold = 0.8
                filterd_points = NearNeighborRemover(distance_threshold=distance_threshold).remove_near_neighbors(
                    pts)  # remove near neighbors (increase distance_threshold to reduce more points)
                if len(filterd_points) < 3:
                    filterd_points = pts
                if len(filterd_points) < 3:
                    continue
                area += area_sub
                points_list.append(filterd_points)
            for pts in points_list:
                mask_pts = []
                for i in range(0, len(pts)):
                    mask_pts.append(pts[i][0]/100)
                    mask_pts.append(pts[i][1]/100)
                export_result[id_str]['masks'].append(mask_pts)
            export_result[id_str]['area'] = area

            lar = np.concatenate(
                (mask*RGB[2], mask*RGB[1], mask*RGB[0]), axis=2)
            zero_mask = zero_mask + lar

        lar_valid = zero_mask > 0
        masked_image = lar_valid*copyed_img
        img_n = masked_image*mask_image_mix_ration + \
            np.clip(zero_mask, 0, 1)*255*(1-mask_image_mix_ration)
        max_p = img_n.max()
        img_n = 255*img_n/max_p
        ret = (~lar_valid*copyed_img)*mask_image_mix_ration + img_n
        ret = ret.astype('uint8')
    else:
        ret = copyed_img

    if 'box' in results_select:
        for nn, (classtext, box) in enumerate(zip(pred_class, boxes)):
            id_str = str(nn)
            x1, y1, x2, y2 = box.long().tolist()

            if prompt_mode == 'categories' or (prompt_mode == 'expression' and 'expression' in results_select):
                if prompt_mode == 'categories':
                    label = ''
                    if 'name' in results_select:
                        label += classtext+'_'
                    if 'score' in results_select:
                        label += str(pred_scores[nn].item())[:4]
                else:
                    label = expressiong

                if len(label) == 0:
                    continue
                height, width, _ = ret.shape

                export_result[id_str]['label'] = label.split('_')[
                    0].replace(' ', '')
                export_result[id_str]['score'] = label.split('_')[1]
                export_result[id_str]['bboxes'] = (
                    (x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height)

                RGB = (COLORS[nn % 12][2]*255, COLORS[nn %
                       12][1]*255, COLORS[nn % 12][0]*255)
                classtext = classtext.replace(' ', '').capitalize()
                score = pred_scores[nn].item()
                if classtext in cate_color_map:
                    RGB = cate_color_map[classtext]
                if classtext == 'Warningsign':
                    classtext = 'Warn'
                elif classtext == 'Signboard':
                    classtext = 'Sign'
                elif classtext == 'Retainingwall':
                    classtext = 'Wall'
                if classtext not in ['Wall', 'Road']:
                    if score < instance_threshold:
                        continue
                label = classtext+':'+label.split('_')[1]
                line_width = max(ret.shape) / 400
                if classtext not in ['Wall', 'Road']:
                    cv2.rectangle(ret, (x1, y1), (x2, y2),
                                  RGB,  math.ceil(line_width))
                FONT = cv2.FONT_HERSHEY_COMPLEX
                label_width, label_height = cv2.getTextSize(label, FONT, min(
                    width, height) * FONT_SCALE, math.ceil(min(width, height) * THICKNESS_SCALE))[0]
                if classtext in ['Wall', 'Road']:
                    if score < semiseg_threshold:
                        continue
                    x = int((x1+x2)/2)
                    y = int((y1+y2)/2)
                    x1, y1 = x, y
                cv2.rectangle(ret, (x1, y1), (x1+label_width, (y1 -
                                                               label_height) - int(height * TEXT_Y_OFFSET_SCALE)), RGB, -1)
                cv2.putText(
                    ret,
                    label,
                    (x1, y1 - int(height * TEXT_Y_OFFSET_SCALE)),
                    fontFace=FONT,
                    fontScale=min(width, height) * FONT_SCALE,
                    thickness=math.ceil(
                        min(width, height) * THICKNESS_SCALE),
                    color=(255, 255, 255),
                )

    ret = ret.astype('uint8')
    return ret, export_result, caption_pred.capitalize()


def visual_prompt_preview(img, prompt_mode):

    copyed_img = img['background'][:, :, :3].copy()

    bbox_list = [scribble2box(layer) for layer in img['layers']]
    zero_mask = np.zeros_like(copyed_img)

    for mask, (box, RGB) in zip(img['layers'], bbox_list):
        if box is None:
            continue

        if prompt_mode == 'box':
            fakemask = np.zeros_like(copyed_img[:, :, 0])
            x1, y1, x2, y2 = box
            fakemask[y1:y2, x1:x2] = 1
            fakemask = fakemask > 0
        elif prompt_mode == 'point':
            fakemask = np.zeros_like(copyed_img[:, :, 0])
            H, W = fakemask.shape
            x1, y1, x2, y2 = box
            center_x, center_y = (x1+x2)//2, (y1+y2)//2
            fakemask[center_y-H//40:center_y+H//40,
                     center_x-W//40:center_x+W//40] = 1
            fakemask = fakemask > 0
        else:
            fakemask = mask[:, :, -1]
            fakemask = fakemask > 0

        mask = fakemask.reshape(fakemask.shape[0], fakemask.shape[1], 1)
        lar = np.concatenate((mask*RGB[0], mask*RGB[1], mask*RGB[2]), axis=2)
        zero_mask = zero_mask + lar

    img_n = copyed_img + np.clip(zero_mask, 0, 255)
    max_p = img_n.max()
    ret = 255*img_n/max_p
    ret = ret.astype('uint8')
    return ret


with gr.Blocks(title='UnsIS',
               theme=gr.themes.Soft(primary_hue="blue")) as demo:
    with gr.Tab("Image task"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image()
                with gr.Row():
                    num_inst_select = gr.Slider(
                        1, 50, value=50, step=1, label="Num of topK", visible=False)
                    base_threshold = gr.Slider(
                        0, 1, value=0.3, label=" Thr_base")
                    instance_threshold = gr.Slider(
                        0, 1, value=0.35, label="Thr_ins")
                    semiseg_threshold = gr.Slider(
                        0, 1, value=0.3, label="Thr_seg")
                    mask_image_mix_ration = gr.Slider(
                        0, 1, value=0.7, label="Brightness Ratio", visible=False)
            with gr.Column():
                image_segment = gr.Image(
                    label="detection and segmentation results", format='png',  show_label=False)
                catption = gr.Text(label='image2caption result',
                                   show_copy_button=True, visible=False)
        with gr.Row():
            image_button = gr.Button("Detect & Segment", variant='primary')
        with gr.Row():
            with gr.Column():
                export_result = gr.Checkbox(
                    label="Export Result", value=False, visible=False)
                model_select = gr.Dropdown(
                    ["GLEE-Lite (R50)", "GLEE-Plus (SwinL)"], value="GLEE-Lite (R50)", multiselect=False, label="Model", visible=False
                )
                with gr.Row():
                    with gr.Column(visible=False):
                        prompt_mode_select = gr.Radio(["categories", "point", "scribble", "box", "expression"],
                                                      label="Prompt", value="categories", info="What kind of prompt do you want to use?")
                        category_select = gr.Dropdown(
                            ["COCO-80", "OBJ365", "Custom-List", "Class-Agnostic"], value="Custom-List", multiselect=False, label="Categories", info="Choose an existing category list or class-agnostic"
                        )
                        custom_category = gr.Textbox(
                            label="Custom Category",
                            info="Input custom category list, seperate by ',' ",
                            lines=1,
                            value="'barrier, building, truck, widebody, car, excavator, machinery, person, warningsign, signboard, fence, rock, road, retainingwall, puddle, slit, rut, cable; truck, person, car"
                        )
                        input_expressiong = gr.Textbox(
                            label="Expression",
                            info="Input any description of an object in the image ",
                            lines=2,
                            value="",
                        )
                    with gr.Group(visible=False):

                        with gr.Accordion("Interactive segmentation usage", open=False):
                            gr.Markdown(
                                'For interactive segmentation:<br />\
                                    1.Draw points, boxes, or scribbles on the canvas for multiclass segmentation; use separate layers for different objects, adding layers with a "+" sign.<br />\
                                    2.Point mode accepts a single point only; multiple points default to the centroid, so use boxes or scribbles for larger objects.<br />\
                                    3.After drawing, click green "√" to preview the prompt visualization; the segmentation mask follows the chosen prompt colors.'
                            )
                        with gr.Accordion("Text based detection usage", open=False):
                            gr.Markdown(
                                'GLEE supports three kind of object perception methods: category list, textual description, and class-agnostic.<br />\
                                1.Select an existing category list from the "Categories" dropdown, like COCO or OBJ365, or customize your own list.<br />\
                                2.Enter arbitrary object name in "Custom Category", or choose the expression model and describe the object in "Expression Textbox" for single object detection only.<br />\
                                3.For class-agnostic mode, choose "Class-Agnostic" from the "Categories" dropdown.'
                            )
                        img_showbox = gr.Image(
                            label="visual prompt area preview")
            with gr.Column(visible=False):
                with gr.Accordion("Try More Visualization Options"):
                    results_select = gr.CheckboxGroup(["box", "mask", "name", "score", "expression"], value=[
                                                      "box", "mask", "name", "score"], label="Shown Results", info="The results shown on image")
        results = gr.JSON(value={}, visible=False)

        def segment_image_example(image_path):
            return segment_image(image_path, prompt_mode_select, category_select, custom_category, input_expressiong,
                                 results_select, num_inst_select, base_threshold, instance_threshold, semiseg_threshold, mask_image_mix_ration, model_select, export_result)
        with gr.Accordion("Results with COCO", open=False):
            results = gr.JSON(value={})
        image_button.click(segment_image, inputs=[img_input, prompt_mode_select, category_select, custom_category, input_expressiong,
                           results_select, num_inst_select, base_threshold, instance_threshold, semiseg_threshold, mask_image_mix_ration, model_select, export_result], outputs=[image_segment, results, catption])
        img_input.change(segment_image, inputs=[img_input, prompt_mode_select, category_select, custom_category, input_expressiong,
                                                results_select, num_inst_select, base_threshold, instance_threshold, semiseg_threshold, mask_image_mix_ration, model_select, export_result], outputs=[image_segment, results, catption])

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0',
                server_port=8601,
                favicon_path='./assets/logo.png',
                allowed_paths=["/"],
                share=True)

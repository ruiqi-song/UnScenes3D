#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-03-26 15:31:15
Description: 
LastEditors: knightdby
LastEditTime: 2025-05-15 18:16:36
FilePath: /UnScenes3D/pipline/uns_label4d/semantic/autoanno_2ddetseg.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-03-26 15:31:15
"""

from gradio_client import Client, file
import pyrootutils
from manifast import *
import math
import json
client = Client("http://0.0.0.0:8601/")

classes_2d_seg = ['background', 'truck', 'widebody', 'car', 'excavator', 'machinery',
                  'person', 'warningsign', 'signboard', 'fence', 'cable',
                  'rock', 'puddle', 'slit', 'rut', 'building',
                  'retainingwall', 'road']
class_to_name_seg = {index: value for index,
                     value in enumerate(classes_2d_seg)}
name_to_class_seg = {v: n for n, v in class_to_name_seg.items()}


def model_infer_glee(path, base_threshold=0.3, instance_threshold=0.35, semiseg_threshold=0.3):
    results = client.predict(
        img=file(path),
        prompt_mode="categories",
        categoryname="Custom-List",
        custom_category="'barrier, building, truck, widebody, car, excavator, machinery, person, warningsign, signboard, fence, rock, road, retainingwall, puddle, slit, rut, cable; truck, person, car",
        expressiong="",
        results_select=["box", "mask", "name", "score"],
        num_inst_select=50,
        base_threshold=base_threshold,
        instance_threshold=instance_threshold,
        semiseg_threshold=semiseg_threshold,
        mask_image_mix_ration=0.7,
        model_selection="GLEE-Lite (R50)",
        export_result=False,
        api_name="/segment_image"
    )
    return results


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


object_3d_frame = 0
object_3d_num = 0


def generat_seg_map(label_path, save_labelid_path):
    global object_3d_num
    global object_3d_frame
    with open(label_path) as f:
        json_cont = json.load(f)
        img_w, img_h = json_cont['image_info']['width'], json_cont['image_info']['height']
        gt_mask = np.zeros((img_h, img_w, 3), dtype='uint8')
        object_3d_frame += 1
        for obj in json_cont['instance']:
            obj_type = obj['obj_type'].lower()
            if obj_type in ['truck', 'widebody', 'car', 'excavator', 'machinery',
                            'person']:
                object_3d_num += 1
            points = obj['mask']
            if obj_type == 'road':
                points = np.array(points).reshape(-1, 1, 2)
                points[:, :, 0] = points[:, :, 0]*img_w
                points[:, :, 1] = points[:, :, 1]*img_h
                points = points.astype(int)
                gt_mask = cv2.fillPoly(
                    gt_mask, [points], name_to_class_seg[obj_type])
        # return
        for obj in json_cont['instance']:
            obj_type = obj['obj_type'].lower()
            points = obj['mask']
            if obj_type == 'retainingwall':
                points = np.array(points).reshape(-1, 1, 2)
                points[:, :, 0] = points[:, :, 0]*img_w
                points[:, :, 1] = points[:, :, 1]*img_h
                points = points.astype(int)
                gt_mask = cv2.fillPoly(
                    gt_mask, [points], name_to_class_seg[obj_type])
        for obj in json_cont['instance']:
            obj_type = obj['obj_type'].lower()
            points = obj['mask']
            if obj_type not in ['backgroundwall', 'retainingwall', 'sky', 'road']:
                points = np.array(points).reshape(-1, 1, 2)
                points[:, :, 0] = points[:, :, 0]*img_w
                points[:, :, 1] = points[:, :, 1]*img_h
                points = points.astype(int)
                gt_mask = cv2.fillPoly(
                    gt_mask, [points], name_to_class_seg[obj_type])
        if not osp.exists(save_labelid_path):
            cv2.imwrite(save_labelid_path, gt_mask[:, :, 0])


file_num = 0
null_num = 0


def run_function(images_files):
    global file_num
    global null_num
    # if len(images_files) < (file_num+10):
    #     print('waiting new data to infer...')
    #     time.sleep(60*10)
    #     null_num += 1
    #     if null_num > 5:
    #         sys.exit(0)
    #     return
    # else:
    #     null_num = 0
    #     file_num = len(images_files)
    for img_path in tqdm(images_files):
        label_2d_path = img_path.replace(
            'images', '../labels/label_2d').replace('.jpg', '.json')
        seg_mask_path = label_2d_path.replace(
            'label_2d', 'label_2d_vis_2').replace('.json', '.png')
        if osp.exists(seg_mask_path):
            continue
        make_path_dirs(seg_mask_path)
        seg_mask, results, _ = model_infer_glee(img_path)
        shutil.copy(seg_mask, seg_mask_path)
        img = cv2.imread(img_path)
        det_info = {}
        det_info['image_info'] = {}
        det_info['image_info']['file_name'] = osp.basename(img_path)
        width = img.shape[1]
        height = img.shape[0]
        det_info['image_info']['width'] = img.shape[1]
        det_info['image_info']['height'] = img.shape[0]
        det_info['image_info']['description'] = "instance contains bbox and mask, bbox: xywhn, mask: [polygon], yolo format "
        # det_info["categories"] = ', '.join(list(cate_18_id2name.values()))
        det_info['instance'] = []

        for rst in results.values():
            for mask in rst['masks']:
                if len(mask) < 4:
                    continue
                area_mask = np.array(mask).reshape(-1, 2)
                area_mask[:, 0] = area_mask[:, 0]*1000
                area_mask[:, 1] = area_mask[:, 1]*1000
                area = cv2.contourArea(area_mask.astype(np.int32))
                # print(rst['label'].capitalize(), area)
                if rst['label'] in ['puddle', 'slit', 'rut'] and area < 5000:
                    continue
                elif rst['label'] in ['building'] and area < 1000:
                    continue
                elif rst['label'] in ['road', 'retainingwall'] and area < 2000:
                    continue

                instance = {}
                instance['obj_type'] = rst['label'].capitalize()
                instance['obj_id'] = -1
                instance['area'] = area/100.
                instance['bbox'] = rst['bboxes']
                instance['mask'] = []
                instance['mask'].append(mask)
                if len(instance['mask']) == 0:
                    continue
                det_info['instance'].append(instance)
        make_path_dirs(label_2d_path)

        with open(label_2d_path, 'w') as outfile:
            json.dump(det_info, outfile, indent=4)

        json_dict = {"version": "4.6.0", "flags": {}, "shapes": [], "imagePath": "", "imageData": None,
                     "imageHeight": 0, "imageWidth": 0}

        json_dict["imagePath"] = '../images/'+osp.basename(img_path)
        json_dict["imageHeight"] = det_info['image_info']['height']
        json_dict["imageWidth"] = det_info['image_info']['width']

        for obj in det_info['instance']:
            img_mark_inf = {"label": "",
                            "line_color":  None,
                            "fill_color": None,
                            "points": [], "shape_type": "polygon", "flags": {}}
            img_mark_inf["label"] = obj['obj_type']
            img_mark_inf["shape_type"] = 'polygon'
            for points in obj['mask']:
                pts = np.array(points).reshape((-1, 2))
                pts[:, 0] = pts[:, 0]*width
                pts[:, 1] = pts[:, 1]*height
                for pt in pts:
                    pt_ = [int(pt[0]), int(pt[1])]
                    img_mark_inf["points"].append(pt_)
            json_dict["shapes"].append(img_mark_inf)
        json_save_path = label_2d_path.replace('label_2d', 'label_me')
        make_path_dirs(json_save_path)
        with open(json_save_path, "w") as out_json:
            json.dump(json_dict, out_json, ensure_ascii=False, indent=2)
        os.makedirs(os.path.dirname(json_save_path.replace(
            'label_2d', 'label_segmap')), exist_ok=True)
        save_labelid_path = label_2d_path.replace(
            'label_2d', 'label_segmap').replace('.json', '.png')
        make_path_dirs(save_labelid_path)
        generat_seg_map(label_2d_path, save_labelid_path)

        # break


if __name__ == '__main__':
    dataset_dir = '/media/knight/disk2knight/htmine_occ'
    samples_img_dir = osp.join(dataset_dir, 'samples/images')
    # samples_img_dir = osp.join(dataset_dir, 'sweeps/images')
    images_files = get_files(samples_img_dir)
    run_function(images_files)

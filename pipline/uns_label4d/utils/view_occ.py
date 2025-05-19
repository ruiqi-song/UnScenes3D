#!/usr/bin/env python3
# coding=utf-8
'''
   brief: 
   Version: v1.0.0
   Author: Baiyong Ding && baiyong.ding@waytous.com
   Date: 2024-11-20 10:31:38
   Description: 
   LastEditors: Baiyong Ding
   LastEditTime: 2024-11-20 10:31:38
   FilePath: /Uni3DScenes_dev_toolkit_/tools/visualization/occ_visual.py
   Copyright 2024  by Inc, All Rights Reserved. 
   2024-11-20 10:31:38
'''
import os
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
from mayavi import mlab
mlab.options.offscreen = True
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))
point_cloud_range = [0, -38.4, -4, 76.8, 38.4, 5.6]
occ_size = [256, 256, 32]
voxel_size = [0.3, 0.3, 0.3]
camera_names = ['CAM_FRONT']
unstructured_road_classname = ['background', 'barrier',
                               'truck', 'widebody', 'car', 'excavator', 'machinery',
                               'pedestrian',
                               'traffic_cone',
                               'muddy',
                               'terrain',
                               'driveable_surface',
                               ]
unstructured_road_classid2name = {index: value for index,
                                  value in enumerate(unstructured_road_classname)}
unstructured_road_classname2id = {v: n for n,
                                  v in unstructured_road_classid2name.items()}
colors = np.array(
    [
        [255, 120, 50, 255],  # barrier              orangey
        [160, 32, 240, 255],  # truck                purple
        [255, 192, 203, 255],  # widebody              pink
        [0, 150, 245, 255],  # car                  blue
        [255, 255, 0, 255],  # excavator                  yellow
        [0, 255, 255, 255],  # machinery cyan
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [139, 137, 137, 255],  # muddy
        [150, 240, 80, 255],  # no driveable_surface            light green
        [255, 0, 255, 255],  # driveable_surface    dark pink
        [139, 137, 137, 255],
        [75,   0,  75, 255],       # sidewalk             dard purple
        [150, 240,  80, 255],       # terrain              light green
        [230, 230, 250, 255],       # manmade              white
        [0, 175,   0, 255],       # vegetation           green
        [0, 255, 127, 255],       # ego car              dark cyan
        [255,  99,  71, 255],
        [0, 191, 255, 255]
    ]
).astype(np.uint8)


def get_files(dir_path, ext=None):
    files = []
    for entry in os.scandir(dir_path):
        if entry.is_file() and (entry.name.endswith(ext) if ext else True):
            files.append(entry.path)
        elif entry.is_dir():
            files.extend(get_files(entry.path, ext))
    files = sorted(files)
    return files


def get_inv_matrix(file, v2c, rect):
    with open(file) as f:
        lines = f.readlines()
        trans = [x for x in filter(lambda s: s.startswith(v2c), lines)][0]
        matrix = [m for m in map(lambda x: float(
            x), trans.strip().split(" ")[1:])]
        matrix = matrix + [0, 0, 0, 1]
        m = np.array(matrix)
        velo_to_cam = m.reshape([4, 4])
        trans = [x for x in filter(lambda s: s.startswith(rect), lines)][0]
        matrix = [m for m in map(lambda x: float(
            x), trans.strip().split(" ")[1:])]
        m = np.array(matrix).reshape(3, 3)
        m = np.concatenate((m, np.expand_dims(np.zeros(3), 1)), axis=1)
        rect = np.concatenate(
            (m, np.expand_dims(np.array([0, 0, 0, 1]), 0)), axis=0)
        m = np.matmul(rect, velo_to_cam)
        m = np.linalg.inv(m)
        return m


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0])  # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1])  # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2])  # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz, indexing='ij')
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid


def draw_nusc_occupancy(
    input_imgs,
    voxels,
    vox_origin,
    voxel_size=0.5,
    grid=None,
    fov_voxel=None,
    pred_lidarseg=None,
    target_lidarseg=None,
    save_folder=None,
    cat_save_file=None,
    cam_positions=None,
    focal_positions=None,
):
    w, h, z = voxels.shape
    grid = grid.astype(np.int32)

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
    voxels_fl = voxels.reshape(-1)
    grid_coords = np.vstack([grid_coords.T, voxels_fl]).T

    grid_coords[grid_coords[:, 3] == 17, 3] = 20
    # w = 248*2
    x_ = 0
    car_vox_range = np.array([
        # [-6, 2],
        # [242, 250],
        [x_, x_+8],
        # [w//2 - 2 - 4, w//2 - 2 + 4],
        [h//2 - 2 - 4, h//2 - 2 + 4],
        [z//2 - 2 - 3, z//2 - 2 + 3]
    ], dtype=np.int32)
    # print(car_vox_range)

    ''' draw the colorful ego-vehicle '''
    car_x = np.arange(car_vox_range[0, 0], car_vox_range[0, 1])
    car_y = np.arange(car_vox_range[1, 0], car_vox_range[1, 1])
    car_z = np.arange(car_vox_range[2, 0], car_vox_range[2, 1])
    car_xx, car_yy, car_zz = np.meshgrid(car_x, car_y, car_z)
    car_label = np.zeros([8, 8, 6], dtype=np.int32)
    car_label[:3, :, :2] = 17
    car_label[3:6, :, :2] = 18
    car_label[6:, :, :2] = 19
    car_label[:3, :, 2:4] = 18
    car_label[3:6, :, 2:4] = 19
    car_label[6:, :, 2:4] = 17
    car_label[:3, :, 4:] = 19
    car_label[3:6, :, 4:] = 17
    car_label[6:, :, 4:] = 18
    car_grid = np.array(
        [car_xx.flatten(), car_yy.flatten(), car_zz.flatten()]).T
    car_indexes = car_grid[:, 0] * h * z + car_grid[:, 1] * z + car_grid[:, 2]
    grid_coords[car_indexes, 3] = car_label.flatten()

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords
    # Remove empty and unknown voxels
    # fov_voxels = np.load(visual_path)

    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
    ]
    # fov_voxels = fov_grid_coords

    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # Draw occupied inside FOV voxels
    voxel_size = sum(voxel_size) / 3
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,  # 16
    )
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    scene = figure.scene
    # mlab.show()

    os.makedirs(save_folder, exist_ok=True)
    visualize_keys = ['CAM_FRONT',  'DRIVING_VIEW', 'BIRD_EYE_VIEW']

    for i in range(3):
        # from six cameras
        if i < 1:
            scene.camera.position = cam_positions[i]
            # - np.array([0.7, 1.3, 0.])
            scene.camera.focal_point = focal_positions[i]
            # -np.array([0.7, 1.3, 0.])
            scene.camera.view_angle = 41
            # scene.camera.view_angle = 35 if i != 3 else 60
            scene.camera.view_up = [0.0, 0.0, 1.0]
            scene.camera.clipping_range = [0.01, 300.]
            scene.camera.compute_view_plane_normal()
            scene.render()

        # bird-eye-view and facing front
        elif i == 1:
            scene.camera.focal_point = [
                -20.4, -0.0,  34.01378558]
            scene.camera.position = [-22.75131739, -0.0,  35.71378558]
            # scene.camera.position = [-30.75131739,  -0.78265103, 16.21378558]
            # scene.camera.focal_point = [-15.25131739,  -0.78265103, 12.21378558]
            scene.camera.view_angle = 40.0
            scene.camera.view_up = [1.0, 0.0, 0.0]
            scene.camera.clipping_range = [0.01, 300.]
            scene.camera.compute_view_plane_normal()
            scene.render()

        # bird-eye-view
        else:
            scene.camera.position = [39.75131739,  0.0, 70.21378558]
            scene.camera.focal_point = [39.75131739,  0.0, 67.21378558]
            scene.camera.view_angle = 60.0
            scene.camera.view_up = [1., 0., 0.]
            scene.camera.clipping_range = [0.01, 400.]
            scene.camera.compute_view_plane_normal()
            scene.render()

        save_file = os.path.join(
            save_folder, '{}.png'.format(visualize_keys[i]))
        mlab.savefig(save_file)
    mlab.close()
    # read rendered images, combine, and create the predictions
    cam_img_size = [480*2, 270*2]
    pred_img_size = [int(1920/3*2), int(1080/3*2)]
    spacing = 10

    cam_w, cam_h = cam_img_size
    pred_w, pred_h = pred_img_size
    result_w = cam_w * 2 + 1 * spacing
    result_h = cam_h * 1 + pred_h + 1 * spacing

    pred_imgs = []
    for cam_name in camera_names:
        pred_img_file = os.path.join(save_folder, '{}.png'.format(cam_name))
        pred_img = Image.open(pred_img_file).resize(
            cam_img_size, Image.BILINEAR)
        pred_imgs.append(pred_img)
    # print(pred_img_size)
    drive_view_occ = Image.open(os.path.join(
        save_folder, 'DRIVING_VIEW.png')).resize(pred_img_size, Image.BILINEAR)
    bev_occ = Image.open(os.path.join(save_folder, 'BIRD_EYE_VIEW.png')).resize(
        pred_img_size, Image.BILINEAR).crop([320, 0, 960, 720])
    # create the output image
    result = Image.new(pred_imgs[0].mode, (result_w, result_h), (0, 0, 0))
    result.paste(input_imgs[0], box=(0, 0))
    result.paste(pred_imgs[0], box=(1*cam_w+1*spacing, 0))
    result.paste(drive_view_occ, box=(0, 1*cam_h+1*spacing))
    result.paste(bev_occ, box=(1*pred_w+1*spacing, 1*cam_h+1*spacing))
    result.save(cat_save_file)
    for i in range(len(input_imgs)):
        input_imgs[i].save(os.path.join(
            save_folder, '{}_ori.png'.format(visualize_keys[i])))


def visual_occ_htmine(visual_path, voxel_in=None, cat_save_file='./results/occ_visual.png'):
    cat_save_file = visual_path.replace(
        'pc_occ', 'pc_occ_view').replace('.npy', '.png')
    fov_voxels = np.load(visual_path)
    fov_voxels[..., 3][fov_voxels[..., 3] == 0] = 255
    voxel = np.zeros(np.array(occ_size))
    voxel[fov_voxels[:, 0].astype(np.int32), fov_voxels[:, 1].astype(
        np.int32), fov_voxels[:, 2].astype(np.int32)] = fov_voxels[:, 3]
    calib_path = visual_path.replace(
        'pc_occ', 'calib').replace('.npy', '.txt')
    matrix = get_inv_matrix(calib_path, "Tr_velo_to_cam", "R0_rect")
    trans = matrix[:3, 3]
    rots = matrix[:3, :3]
    cam2lidar = np.repeat(np.eye(4)[np.newaxis], repeats=1, axis=0)
    cam2lidar[:, :3, :3] = rots
    cam2lidar[:, :3, -1] = trans

    img_canvas = []
    img_path = visual_path.replace(
        'pc_occ', 'camera_1').replace('.npy', '.jpg')
    cam_img_size = [480*2, 270*2]
    img = Image.open(img_path)
    img = img.resize(cam_img_size, Image.BILINEAR)
    img_canvas.append(img)
    cam_positions = cam2lidar @ np.array([0., 0., 0., 1.])
    cam_positions = cam_positions[:, :3]
    constant_f = 0.0055
    focal_positions = cam2lidar @ np.array([0., 0., constant_f, 1.])
    focal_positions = focal_positions[:, :3]
    sample_token = os.path.basename(visual_path)[:-4]
    save_folder = os.path.dirname(cat_save_file)+'/assets_occ'
    os.makedirs(save_folder, exist_ok=True)

    draw_nusc_occupancy(
        input_imgs=img_canvas,
        voxels=voxel,
        vox_origin=np.array(point_cloud_range[:3]),
        voxel_size=np.array(voxel_size),
        grid=np.array(occ_size),
        pred_lidarseg=None,
        target_lidarseg=None,
        fov_voxel=None,
        save_folder=save_folder,
        cat_save_file=cat_save_file,
        cam_positions=cam_positions,
        focal_positions=focal_positions,
    )


if __name__ == '__main__':
    DATA_DIR = './data/raw_data/scene_00000'
    occ_paths = get_files(DATA_DIR+'/pc_occ', '.npy')
    # random.shuffle(occ_paths)
    for occ_path in tqdm(occ_paths):
        visual_occ_htmine(occ_path)
        # break

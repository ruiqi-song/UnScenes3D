# pipline

The UnScene3D pipeline includes :

1. uns_is — an instance segmentation server built with Gradio;
2. uns_label4d — scripts for generating 4D labels;
3. uns_2kitti — a conversion tool that transforms UnScene3D data into KITTI format for model training.

The pipline framework is shown below:

<p align="center"><img src=../assets/pipline/pipline.png>
</p>

## 1. uns_is

1. Install environment following [GLEE INSTALL](https://github.com/FoundationVision/GLEE/blob/main/assets/INSTALL.md);
2. Download the pre-trained model [here](https://github.com/ruiqi-song/UnScenes3D/releases/download/uns_is/uns_is_model_weights.pth) and put it under `weights/unscene_2d`;
3. Download the clip_vit_base_patch32 model [here](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/GLEE/clip_vit_base_patch32/pytorch_model.bin) and put it under `weights/clip_vit_base_patch32`;
4. Run the following command to start the server: `python pipline/uns_is/app.py`;
5. Open the link in the console to start instance segmentation:
<p align="center"><img src=../assets/pipline/app.png width="600" >
</p>

## 2. uns_label4d

1. Run `python pipeline/uns_label4d/gen_odom_fine.py` to fine-tune the odometry using the **KISS-ICP** algorithm.
2. Generate instance segmentation labels by running:`python pipeline/uns_label4d/gen_label_2d.py`
3. Generate occupancy labels:
   - Run `python pipeline/uns_label4d/src/static_obs_builder.py` and `python pipeline/uns_label4d/src/static_obs_invis.py` to generate static obstacles. These scripts rely on instance segmentation labels and the pt2pixel project to calculate static obstacle bounding boxes.
   - Run `python pipeline/uns_label4d/gen_pc_semantic.py` to generate the semantic point cloud, based on both static and dynamic obstacle bounding boxes.
   - Run `python pipeline/uns_label4d/gen_pclabel_occ.py` to generate occupancy labels, following the SurroundOcc labeling strategy.
4. Generate depth labels by running:`python pipeline/uns_label4d/gen_label_depth.py`. This process uses the local map cloud to project depth onto images.
5. Generate elevation labels by running:
   `python pipeline/uns_label4d/gen_label_height.py`
   Similar to depth labeling, this also uses the local map cloud projection.
6. Generate caption and vehicle information labels:

   - Run `python pipeline/uns_label4d/gen_label_caption.py` to generate caption labels based on instance segmentation.
   - Run `python pipeline/uns_label4d/gen_label_vehinfo.py` to generate vehicle information labels.

## 3. uns_2kitti

1. convert the data to kitti format by running `python pipline/uns_2kitti/occ_pred.py` for 3D semantic occupancy prediction task, which is compatible with the data format required by the [OccFormer](https://github.com/zhangyp15/OccFormer) project.
2. convert the data to kitti format by running `python pipline/uns_2kitti/depth_pred.py` for depth estimation task and road surface elevation reconstruction task, which is compatible with the data format required by the [mmdepth](https://github.com/RuijieZhu94/mmdepth) project.

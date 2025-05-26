# pipline

The UnScenes3D pipeline consists of the following components:

- uns_is — An instance segmentation server built with Gradio.
- uns_label4d — A set of scripts for generating 4D labels.
- uns_2kitti — A data conversion tool that transforms UnScenes3D dataset into KITTI format for model training.

The pipeline workflow is illustrated below.

   <p align="center"><img src=../assets/pipline/pipline.png>
   </p>

## 0. prepare

1. Download the UnScenes3D raw data from [Releases](https://github.com/ruiqi-song/UnScenes3D/releases/download/unscenes-mini/raw_data.zip) section(unscenes-mini->raw_data.zip);
2. Extract the contents to `./data/raw_data`, ensuring the structure is as follows:

   ```
   ./data/raw_data/
   ├── scene_00000
   │   ├── calib
   │   ├── camera_1
   │   ├── ego_pose
   │   ├── label_3d
   │   └── lidar_1
   └── scene_info.json
   ```

## 1. uns_is – Instance Segmentation Server

1. Set up the environment by following the instructions in [GLEE INSTALL](https://github.com/FoundationVision/GLEE/blob/main/assets/INSTALL.md);
2. Download the [pre-trained model ](https://github.com/ruiqi-song/UnScenes3D/releases/download/uns_is/uns_is_model_weights.pth) and place it in `weights/unscene_2d`;
3. Download the [clip_vit_base_patch32 model](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/GLEE/clip_vit_base_patch32/pytorch_model.bin) and place it in `weights/clip_vit_base_patch32`;
4. Launch the segmentation server:

   ```bash
   python pipline/uns_is/app.py
   ```

5. Open the link shown in the console to interact with the instance segmentation UI.
<p align="center"><img src=../assets/pipline/app.png width="600" >
</p>

## 2. uns_label4d – Label Generation Scripts

1.  Run the following script to refine odometry using **KISS-ICP**:

    ```bash
    python pipeline/uns_label4d/src/gen_odom_fine.py
    ```

2.  Generate 2D instance segmentation labels:

    ```bash
    python pipeline/uns_label4d/src/gen_label_2d.py
    ```

3.  Generate occupancy labels:
    - Build and run static obstacle generator:
      ```bash
      cd pipeline/uns_label4d
      catkin_make
      source devel/setup.bash
      rosrun uns_label4d obs_4d_builder
      python pipeline/uns_label4d/src/static_obs_builder.py
      ```
    - Generate semantic point clouds:
      ```bash
      python pipeline/uns_label4d/label_4d/gen_pc_semantic.py
      ```
    - Generate occupancy labels (SurroundOcc style):
      ```bash
      python pipeline/uns_label4d/label_4d/gen_pclabel_occ.py
      ```
    - Visualize occupancy labels:
      ```bash
      python pipline/uns_label4d/utils/visual_occ.py
      ```

   <p align="center">
   <img src=../assets/pipline/label_4d.png width="42%" style="display: inline-block; margin-right: 2%;" />
   <img src=../assets/pipline/occ_label.png width="37%" style="display: inline-block; margin-right: 2%;" />

4.  Generate Depth & Elevation Labels

    - Generate depth labels using local map cloud projection:

      ```bash
      python pipeline/uns_label4d/label_4d/gen_label_depth.py
      ```

    - Generate elevation (height) labels:

      ```bash
      python pipeline/uns_label4d/label_4d/gen_label_height.py
      ```

    <p align="center">
    <img src=../assets/pipline/label_depth.png width="47%" style="display: inline-block; margin-right: 2%;" />
     <img src=../assets/pipline/label_elevation.png width="31%" style="display: inline-block; margin-right: 2%;" />

5.  Generate Caption & Vehicle Information Labels:

    - Generate caption labels based on instance segmentation:
      ```bash
      python pipeline/uns_label4d/label_4d/gen_label_caption.py
      ```
    - Generate vehicle info labels:
      ```bash
      python pipeline/uns_label4d/label_4d/gen_label_vehinfo.py
      ```

## 3. uns_2kitti – Convert to KITTI Format

1. 3D Semantic Occupancy Prediction

   Convert data into KITTI-compatible format for [OccFormer](https://github.com/zhangyp15/OccFormer) :

   ```bash
   python pipline/uns_2kitti/occ_pred.py
   ```

2. Depth Estimation & Elevation Reconstruction

   Convert data into KITTI-compatible format for [mmdepth](https://github.com/RuijieZhu94/mmdepth):

   ```bash
   python pipline/uns_2kitti/depth_pred.py
   ```

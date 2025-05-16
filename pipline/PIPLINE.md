# pipline

The UnScene3D pipeline includes:
(1) uns_is — an instance segmentation server built with Gradio;
(2) uns_label4d — scripts for generating 4D labels;
(3) uns_2kitti — a conversion tool that transforms UnScene3D data into KITTI format for model training.

## uns_is

1. Install environment following [GLEE INSTALL](https://github.com/FoundationVision/GLEE/blob/main/assets/INSTALL.md);
2. Download the pre-trained model [here](https://drive.google.com/file/d/1rYNKqxBjg937w6lGmVZ_M2fkUHWuAOX/view?usp=sharing) and put it under `weights`;
3. Run the following command to start the server: `python pipline/uns_is/app.py`;
4. Open the link in the console to start instance segmentation:
   <img src=../assets/app.png>

## uns_label4d

1. generate 2d label by running `python pipline/uns_label4d/gen_label_2d.py`;
2. generate depth label by running `python pipline/uns_label4d/gen_label_depth.py`;
3. generate elevation label by running `python pipline/uns_label4d/gen_label_height.py`;
4. generate occupancy label by :
   - run `python pipline/uns_label4d/gen_odom_fine.py` to fine tune odom;
   - run `python pipline/uns_label4d/gen_pc_semantic.py` to generate semantic point cloud and bbox;
   - run `python pipline/uns_label4d/gen_pclabel_occ.py` to generate occupancy label;
5. generate caption label and vehicle label by running `python pipline/uns_label4d/gen_label_caption.py` and `python pipline/uns_label4d/gen_label_vehinfo.py` respectively;

## uns_2kitti

1. convert the data to kitti format by running `python pipline/uns_2kitti/occ_pred.py` for 3D semantic occupancy prediction task, which is compatible with the data format required by the [OccFormer](https://github.com/zhangyp15/OccFormer) project.
2. convert the data to kitti format by running `python pipline/uns_2kitti/depth_pred.py` for depth estimation task and road surface elevation reconstruction task, which is compatible with the data format required by the [mmdepth](https://github.com/RuijieZhu94/mmdepth) project.

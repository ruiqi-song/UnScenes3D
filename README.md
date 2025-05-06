# Scene as Occupancy and Reconstruction: A Comprehensive Dataset for Unstructured Scene Understanding
### [Paper](https://arxiv.org/pdf/2311.12754)  | [Project Page](https://github.com/ruiqi-song/UnScene-AutoDrive) 

## Overview
<p align="center">
<img src=./assets/overview.png alt="Description" width="600"/>
</p>

> Scene as Occupancy and Reconstruction: A Comprehensive Dataset for Unstructured Scene Understanding

## News
- **[2025/5/10]** UnScenes3D Dataset Release

## Demo

### Trained using only video sequences and poses:

<p align="center">
<img src=./assets/demo.gif.png>
</p>

## Data Pipeline
<p align="center">
<img src=./assets/framework.png alt="Description" width="600"/>
</p>

- We propose a novel 3D semantic occupancy prediction framework that improves the robustness of prediction in unstructured scenes. Bidirectional supervision for cross-modal feature alignment mechanism and detail-aware 3D Gaussian Splatting auxiliary supervision mechanism are proposed to enhance the capability of cross-modal fusion and long-tailed class prediction in unstructured scene, respectively

- Our UnsOcc outperforms the newest and best method L2Occ by 58.7% on UnScenes3D and is the first 3D semantic occupancy prediction work in unstructured scenes.

## Dataset organization
```
Dataset/
├── calibs/                  # Calibration information for sensors
├── images/                  # Synchronized frame image data
│   ├── 1689903584.278848.jpg
│   └── ...
├── clouds/                  # HAP-synchronized frame point cloud data
│   ├── 1689903584.278848.bin
│   └── ...
├── occ/                     # 3D semantic occupancy prediction labels
├── elevation/               # Road elevation labels
├── depths/                  # Depth estimation labels
├── imagesets/               # Dataset splits for training, validation, and testing
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── localmap_clouds/         # Dense point cloud map of local environment
├── vehicle_infos/           # Ego vehicle's pose, speed, and acceleration information
└── image_caption/           # Language-based scene descriptions
```


### Description of Each Directory
<p align="center">
<img src=./assets/dir_desc.png>
</p>

## Dataset Stastic
<img src=./assets/stastic.png>

## Results
# Occupancy baseline
<img src=./assets/occ_nus.png>

# Depth baseline
<img src=./assets/depth_nus.png>



### Visualization

Follow detailed instructions in [Visualization](docs/visualization.md).



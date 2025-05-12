# Unstructured Scene Understanding
[Project Page](https://github.com/ruiqi-song/UnScene-AutoDrive) 
<h2 align="center">Scene as Occupancy and Reconstruction</h2>

[//]: # (<p align="center">)

[//]: # (  <a href="https://github.com/knightdby">)

[//]: # (    <img src="https://img.shields.io/badge/Scene%20as%20Occupancy%20and%20Reconstruction-Core%20Member-607d8b?style=flat-square&logo=github&logoColor=white" height="25px"/>)

[//]: # (  </a>)

[//]: # (</p>)

[//]: # (<h2 align="center">🧑‍💻 Project Members</h2>)

[//]: # (<p align="center">)

[//]: # (  <a href="https://github.com/knightdby">)

[//]: # (    <img src="https://img.shields.io/badge/Baiyong%20Ding-Core%20Member-607d8b?style=flat-square&logo=github&logoColor=white" height="25px"/>)

[//]: # (  </a>)

[//]: # (  <a href="https://github.com/ruiqi-song">)

[//]: # (    <img src="https://img.shields.io/badge/Ruiqi%20Song-Project%20Leader-4caf50?style=flat-square&logo=github&logoColor=white" height="25px"/>)

[//]: # (  </a>)

[//]: # (</p>)


<p align="center">
  <img src=./assets/road_rec_01.gif width="48%" style="display: inline-block; margin-right: 2%;" />
  <img src=./assets/road_rec_02.gif width="48%" style="display: inline-block;" />
</p>

## Overview
<p align="center">
<img src=./assets/overview.png alt="Description" width="600"/>
</p>

> we investigate unstructured scene understanding through 3D semantic occupancy prediction, which is used to detect irregular obstacles in unstructured scenes, and road surface elevation reconstruction, which characterizes the bumpy and uneven conditions of road surfaces. The dataset provides detailed annotations for 3D semantic occupancy prediction and road surface elevation reconstruction, offering a comprehensive representation of unstructured scenes. In addition, trajectory and speed planning information is provided to explore the relationship between perception and planning in unstructured scenes. Natural language descriptions of scenes are also provided to explore the interpretability of autonomous driving decision-making.

## News
- **[2025/5/10]** UnScenes3D Dataset v1.0 Released

## Data Pipeline
<p align="center">
<img src=./assets/framework.png alt="Description" width="600"/>
</p>

> Dataset construction framework and future outlook: (a) Data processing. (b) Data label visualization. (c) Scene text description. (d) Future work outlook.

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

## Tools

### Processing

- generate depth label from localmap_clouds by [this](tools/processing/gengert_depth_label.py). 
- generate elevation label from localmap_clouds by [this](tools/processing/gengert_elevation_label.py).

### Visualization

- [visualize 3D semantic occupancy label.](tools/visualization/occ_visual.py)

## Technical Validation
### Tasks

1. [Restructure the data into the SemanticKITTI dataset structure](tools/tasks/occ_prediction.py) for 3D semantic occupancy prediction task, which is compatible with the data format required by the [OccFormer](https://github.com/zhangyp15/OccFormer) project.
2. [Restructure the data](tools/tasks/depth_elevation_estimation.py) for depth estimation task and road surface elevation reconstruction task, which is compatible with the data format required by the [mmdepth](https://github.com/RuijieZhu94/mmdepth) project.


### 3D Semantic Occupancy Prediction
<img src=./assets/occ_nus.png>

### Road Elevation Reconstruction
<img src=./assets/depth_nus.png>


## Acknowledgement

Many thanks to these excellent open source projects:

- [MonoScene](https://github.com/astra-vision/MonoScene)
- [TPVFormer](https://github.com/wzzheng/TPVFormer)
- [OccFormer](https://github.com/zhangyp15/OccFormer)
- [SurroundOcc](https://github.com/weiyithu/SurroundOcc)
- [CGFormer](https://github.com/pkqbajng/CGFormer)
- [Co-Occ](https://github.com/Rorisis/Co-Occ)
- [L2COcc](https://github.com/StudyingFuFu/L2COcc)
- [mmdepth](https://github.com/RuijieZhu94/mmdepth)


<h2 align="center">🤝 Collaborators</h2>

<p align="center">
  <a href="https://www.ia.cas.cn/"><img src="https://img.shields.io/badge/CASIA-blue?style=flat-square&logo=government&logoColor=white" height="25px"/></a>
  <a href="https://www.waytous.cn/"><img src="https://img.shields.io/badge/Waytous-ff9800?style=flat-square&logo=academia&logoColor=white" height="25px"/></a>
  <a href="https://www.tongji.edu.cn/"><img src="https://img.shields.io/badge/Tongji%20University-005eff?style=flat-square&logo=briefcase&logoColor=white" height="25px"/></a>
</p>



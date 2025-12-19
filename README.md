# Seeing Through Fog Dataloader
## Introduction

The **Seeing Through Fog (STF)** dataset is a large-scale, multimodal autonomous driving dataset designed to study robust perception under adverse weather conditions, particularly dense fog. Unlike conventional driving datasets captured in clear weather, STF focuses on degraded visibility scenarios where camera-based perception systems struggle.

To address this challenge, the dataset provides synchronized multi-sensor data, enabling research on sensor fusion, domain robustness, and adverse-weather perception.

## Dataset Overview

The Seeing Through Fog dataset includes the following sensor modalities:

- RGB Cameras – High-resolution visual data affected by fog density

- Gated Near-Infrared (NIR) Cameras – Enhanced visibility through fog

- LiDAR – Accurate 3D geometry independent of lighting conditions

- Radar – Robust long-range sensing in adverse weather

- Calibration & Synchronization Data – Precise spatial-temporal alignment across sensors

The dataset is collected in real-world fog conditions with varying visibility ranges, enabling realistic evaluation of perception systems.

## Annotations

The dataset provides 3D object annotations in the vehicle coordinate frame, including:

- Object categories such as:

  - 'PassengerCar'

  - 'Pedestrian'

  - 'Cyclist'

  - 'Obstacle'

  - 'PassengerCar_is_group'

- 3D bounding boxes (position, dimensions, orientation)

- 2D image bounding boxes

- Visibility and occlusion indicators

- Annotations follow a KITTI-style format, making the dataset compatible with existing 3D detection and sensor fusion pipelines.

## Research Applications

- The Seeing Through Fog dataset is widely used for:

- Multimodal sensor fusion (camera + LiDAR + radar)

- Robust 3D object detection

- Adverse weather perception

- Cross-modal learning and domain adaptation

- Autonomous driving in low-visibility environments

## Purpose of This Repository

This repository provides a custom dataloader and preprocessing pipeline for the Seeing Through Fog dataset, designed to integrate with 3D detection frameworks such as OpenPCDet. It enables efficient loading, augmentation, and preparation of STF data for training and evaluation of multimodal perception models. 

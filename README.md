
# VoxDet: Voxel Learning for Novel Instance Detection

### NeurIPS'23 **SpotLight**




## Abstract

Detecting unseen instances based on multi-view templates is a challenging problem due to its open-world nature. Traditional methodologies, which primarily rely on 2D representations and matching techniques, are often inadequate in handling pose variations and occlusions. To solve this, we introduce VoxDet, a pioneer 3D geometry-aware framework that fully utilizes the strong 3D voxel representation and reliable voxel matching mechanism. VoxDet first ingeniously proposes a template voxel aggregation (TVA) module, effectively transforming multi-view 2D images into 3D voxel features.  By leveraging associated camera poses, these features are aggregated into a compact 3D template voxel. In novel instance detection, this voxel representation demonstrates heightened resilience to occlusion and pose variations. We also discover that a 3D reconstruction objective helps to pre-train the 2D-3D mapping in TVA.  Second, to quickly align with the template voxel, VoxDet incorporates a Query Voxel Matching (QVM) module. The 2D queries are first converted into their voxel representation with the learned 2D-3D mapping. We find that since the 3D voxel representations encode the geometry, we can first estimate the relative rotation and then compare the aligned voxels, leading to improved accuracy and efficiency. Exhaustive experiments are conducted on the demanding LineMod-Occlusion, YCB-video, and the newly built RoboTools benchmarks, where VoxDet outperforms various 2D baselines remarkably with $\mathbf{20}\%$ higher recall and faster speed. To the best of our knowledge, VoxDet is the first to incorporate implicit 3D knowledge for 2D tasks.


## Requirements

This repo is tested under Python 3.7, PyTorch 1.7.0, Cuda 11.0, and mmcv==1.2.5.




## Installation

This repo is built based on [mmdetection](https://github.com/open-mmlab/mmdetection). 

For evaluation, you also need [bop_toolkit](https://mega.nz/file/BAEj3TgS#yzwX2AHUg9CtCsmDV17rxVkmFhw4mh34y6gvQ3FDS4E)

You can use the following commands to create conda env with related dependencies.
```shell
conda create -n voxdet python=3.7 -y
conda activate voxdet
conda install pytorch=1.7.0 torchvision cudatoolkit=11.0 -c pytorch -y
pip install mmcv-full==1.2.7
pip install -r requirements.txt
pip install -v -e . 

cd ..
cd bop_toolkit
pip install -e .
```



## Prepare datasets

We provide the processed [OWID](https://mega.nz/file/pUlgQa7Z#Qcmj0zh7gUXszeeVPqMLQVYtkknad9_gzqsNQwss6kY), [LM-O](https://mega.nz/file/pUlgQa7Z#Qcmj0zh7gUXszeeVPqMLQVYtkknad9_gzqsNQwss6kY), [YCB-V](https://mega.nz/file/pUlgQa7Z#Qcmj0zh7gUXszeeVPqMLQVYtkknad9_gzqsNQwss6kY) and [RoboTools](https://mega.nz/file/AB82EJwZ#76mVyk-L3cGRJGX7-KDUBcePNE1o1O96G4F58b5PxGI) to reproduce the evaluation.

You can download them and creat data structure like this:

```shell
VoxDet
├── mmdet
├── tools
├── configs
├── data
│   ├── BOP
│   │   ├── lmo
|   |   |   ├── test
|   |   |   ├── test_video
│   │   ├── ycbv
│   │   ├── RoboTools
│   ├── OWID
│   │   ├── P1
│   │   ├── P2
```



## Testing

Our trained models and raw results are available for download here. 

Place it under `outputs/` and run the following commands to test VoxDet on LM-O and YCB-V datasets.

```shell
bash tools/test.sh
```

By default, the script will only calculate results from the raw `.pkl` files, to run VoxDet, you need to change the output file name in the command like

```shell
# lmo
python3 tools/test.py --config configs/voxdet/${CONFIG}.py --out outputs/$OUT_DIR/lmo1.pkl \
--checkpoint outputs/VoxDet_p2_2/model_final.pth >> outputs/$OUT_DIR/lmo1.txt
```

The results will be shown in the `.txt` file.



## Training

Our training set OWID will be released upon acceptance, while we provide the code and script here:

```shell
# Single-GPU training for the reconstruction stage
bash tools/train.sh

# Multi-GPU training for the base detection, this should already produce the results close to table 1
bash tools/train_dist.sh

# Optional, use ground truth rotation for supervision for (slightly) better result 
bash tools/train_dist_2.sh

```

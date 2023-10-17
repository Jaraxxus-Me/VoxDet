
# VoxDet: Voxel Learning for Novel Instance Detection

### Bowen Li, Jiashun Wang, Yaoyu Hu, Chen Wang, and Sebastian Scherer

### NeurIPS'23 :star2: SpotLight :star2:




## Abstract

Detecting unseen instances based on multi-view templates is a challenging problem due to its open-world nature. Traditional methodologies, which primarily rely on $2 \mathrm{D}$ representations and matching techniques, are often inadequate in handling pose variations and occlusions. To solve this, we introduce VoxDet, a pioneer 3D geometry-aware framework that fully utilizes the strong 3D voxel representation and reliable voxel matching mechanism. VoxDet first ingeniously proposes template voxel aggregation (TVA) module, effectively transforming multi-view 2D images into 3D voxel features. By leveraging associated camera poses, these features are aggregated into a compact 3D template voxel. In novel instance detection, this voxel representation demonstrates heightened resilience to occlusion and pose variations. We also discover that a $3 \mathrm{D}$ reconstruction objective helps to pre-train the 2D-3D mapping in TVA. Second, to quickly align with the template voxel, VoxDet incorporates a Query Voxel Matching (QVM) module. The 2D queries are first converted into their voxel representation with the learned 2D-3D mapping. We find that since the 3D voxel representations encode the geometry, we can first estimate the relative rotation and then compare the aligned voxels, leading to improved accuracy and efficiency. In addition to method, we also introduce the first instance detection benchmark, RoboTools, where 20 unique instances are video-recorded with camera extrinsic. RoboTools also provides 24 challenging cluttered scenarios with more than $9 \mathrm{k}$ box annotations. Exhaustive experiments are conducted on the demanding LineMod-Occlusion, YCB-video, and RoboTools benchmarks, where VoxDet outperforms various 2D baselines remarkably with faster speed. To the best of our knowledge, VoxDet is the first to incorporate implicit 3D knowledge for 2D novel instance detection tasks.


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

We provide the processed [OWID](https://drive.google.com/file/d/1sRHaVd4exOmGqFUVT6JKUzEOrDeHmlbT/view?usp=sharing), [LM-O](https://drive.google.com/file/d/1cY8gWF6t0IhEa0nLPVWfHMcPlfTNFPwe/view?usp=sharing), [YCB-V](https://drive.google.com/file/d/1JpixHE9DN-W-BVFkVC12qss0CUu9VA8y/view?usp=sharing) and [RoboTools](https://drive.google.com/file/d/1kXR-Z-sJlTnWy3HRGWAcV6_IIJgRHbD6/view?usp=sharing) to reproduce the evaluation.

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

You can also compile your custom instance detection dataset using [this toolkit](https://github.com/Jaraxxus-Me/OWID-toolkit.git), it is very useful :)


## Testing

Our trained [models and raw results](https://drive.google.com/file/d/1VrXcT6tQwhR0zDlANribjcyAritFqKn7/view?usp=sharing) for all the stages are available for download. 

Place it under `outputs/` and run the following commands to test VoxDet on LM-O and YCB-V datasets.

```shell
bash tools/test.sh
```

By default, the script will only calculate results from the raw `.pkl` files, to actually run VoxDet, you need to change the output file name in the command like

```shell
# lmo
python3 tools/test.py --config configs/voxdet/${CONFIG}.py --out outputs/$OUT_DIR/lmo1.pkl \
--checkpoint outputs/VoxDet_p2_2/iter_100.pth >> outputs/$OUT_DIR/lmo1.txt
```

The results will be shown in the `.txt` file.



## Training

Our training set OWID will be released upon acceptance, while we provide the code and script here:

```shell
# Single-GPU training for the reconstruction stage
bash tools/train.sh

# Multi-GPU training for the base detection, this should already produce the results close to table 1
bash tools/train_dist.sh

# Optional, use ground truth rotation for supervision for (slightly) better result, see table 4 for details
bash tools/train_dist_2.sh

```

## Reference
If our work inspires your research, please cite us as:

```
@INPROCEEDINGS{Li2023iccv,       
	author={Li, Bowen and Wang, Jiashun and Hu, Yaoyu and Wang, Chen and Scherer, Sebastian},   
	booktitle={Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)}, 
	title={{VoxDet: Voxel Learning for Novel Instance Detection}},
	year={2023},
	volume={},
	number={}
}
```

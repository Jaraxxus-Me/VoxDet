export CUDA_VISIBLE_DEVICES=0
OUT_DIR=VoxDet_p2_c
CONFIG=VoxDet_test_c

for iter in 5626 11251 16876 22501 33751
do
# lmo
# python3 tools/test.py --config configs/voxdet/${CONFIG}.py --out outputs/$OUT_DIR/lmo_${iter}.pkl \
# --checkpoint outputs/$OUT_DIR/iter_${iter}.pth >> outputs/$OUT_DIR/lmo_${iter}.txt
# YCB-V
# python3 tools/test.py --config configs/voxdet/${CONFIG}.py --out outputs/$OUT_DIR/ycbv_${iter}.pkl \
# --checkpoint outputs/$OUT_DIR/iter_${iter}.pth \
# --cfg-options dataset='ycbv' data.test.ann_file="data/BOP/ycbv/test/scene_gt_coco_all.json" \
# data.test.p1_path="data/BOP/ycbv/test_video/" \
# data.test.img_prefix="data/BOP/ycbv/test" \
# model.D=54 >> outputs/$OUT_DIR/ycbv_${iter}.txt
# roboTools, this dataset will be released upon acceptance
python3 tools/test.py --config configs/voxdet/${CONFIG}.py --out outputs/$OUT_DIR/robo_${iter}.pkl \
--checkpoint outputs/$OUT_DIR/iter_${iter}.pth \
--cfg-options dataset='RoboTools' data.test.ann_file="data/BOP/RoboTools/test/scene_gt_coco_all.json" \
data.test.p1_path="data/BOP/RoboTools/test_video/" \
data.test.img_prefix="data/BOP/RoboTools/test"  >> outputs/$OUT_DIR/robo_${iter}.txt
done

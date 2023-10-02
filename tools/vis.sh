export CUDA_VISIBLE_DEVICES=0
OUT_DIR=supp_vis/vs_dtoid

# lmo
python3 tools/vis.py --out1 work_dirs/VoxDet_p2_2/bop_lmo_60800.pkl \
--out2 work_dirs/DTOID/bop_lmo.pkl \
--show-dir $OUT_DIR/lmo
# YCB-V
python3 tools/vis.py --out1 work_dirs/VoxDet_p2_2/bop_ycb_60800.pkl \
--out2 work_dirs/DTOID/bop_ycbv.pkl \
--show-dir $OUT_DIR/ycbv \
--cfg-options dataset='ycbv' data.test.ann_file="data/BOP/ycbv/test/scene_gt_coco_all.json" \
data.test.p1_path="data/BOP/ycbv/test_video/" \
data.test.img_prefix="data/BOP/ycbv/test"
# Robotools
python3 tools/vis.py --out1 work_dirs/VoxDet_p2_2/bop_robo_60800.pkl \
--out2 work_dirs/DTOID/bop_robo.pkl \
--show-dir $OUT_DIR/robotools \
--cfg-options dataset='RoboTools' data.test.ann_file="data/BOP/RoboTools/test/scene_gt_coco_all.json" \
data.test.p1_path="data/BOP/RoboTools/test_video/" \
data.test.img_prefix="data/BOP/RoboTools/test"

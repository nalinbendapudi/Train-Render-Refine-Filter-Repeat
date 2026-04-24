DATA_DIR=../../MyData/Mip-Nerf-360/bonsai/
DATA_FACTOR=4
TEST_FRACTION=0.9
INIT_TYPE=sfm
COMMON_CKPT=5k

RESULT_PARENT_DIR="examples/scene_results/${DATA_DIR#*MyData/}/test_fraction_${TEST_FRACTION}/init_${INIT_TYPE}/\
common_ckpt_${COMMON_CKPT}/"

## Train and Retrain

CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
--data_dir ${DATA_DIR} \
--data_factor ${DATA_FACTOR} \
--result_dir ${RESULT_PARENT_DIR}/train_and_retrain/ \
--no-normalize-world-space \
--test_fraction ${TEST_FRACTION} \
--init_type ${INIT_TYPE}


## Refine and Retrain

CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
--data_dir ${DATA_DIR} \
--data_factor ${DATA_FACTOR} \
--result_dir ${RESULT_PARENT_DIR}/novel_effect_1/refine_and_retrain/ \
--no-normalize-world-space \
--test_fraction ${TEST_FRACTION} \
--init_type ${INIT_TYPE} \
--ckpt ${RESULT_PARENT_DIR}/train_and_retrain/ckpts/ckpt_4999_rank0.pt


## Render and Retrain

CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
--data_dir ${DATA_DIR} \
--data_factor ${DATA_FACTOR} \
--result_dir ${RESULT_PARENT_DIR}/novel_effect_1/render_and_retrain/ \
--no-normalize-world-space \
--test_fraction ${TEST_FRACTION} \
--init_type ${INIT_TYPE} \
--ckpt ${RESULT_PARENT_DIR}/train_and_retrain/ckpts/ckpt_4999_rank0.pt \
--novel_views_path ${RESULT_PARENT_DIR}/novel_effect_1/refine_and_retrain/renders/ \
--render_only

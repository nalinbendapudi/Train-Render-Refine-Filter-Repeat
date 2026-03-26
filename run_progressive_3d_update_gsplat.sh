SCENE_ID=ums_lab
DATA=examples/scene_data/imgs_and_colmap/${SCENE_ID}
DATA_FACTOR=1
CKPT_PATH=examples/scene_data/initial_gsplat_ckpts/${SCENE_ID}/ckpts/ckpt_29999_rank0.pt
OUTPUT_DIR=examples/scene_results/${SCENE_ID}

CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${OUTPUT_DIR} \
    --no-normalize-world-space \
    --test_every 8 \
    --ckpt ${CKPT_PATH}

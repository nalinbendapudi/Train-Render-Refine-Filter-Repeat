SCENES=(
    bicycle
    bonsai
    counter
    flowers
    garden
    kitchen
    stump
)

DATA_FACTOR=4
TEST_FRACTION=0.9
INIT_TYPE=sfm
COMMON_CKPT=5k

for SCENE in "${SCENES[@]}"
do
    echo "Running scene: ${SCENE}"

    DATA_DIR=../../MyData/Mip-Nerf-360/${SCENE}/
    RESULT_PARENT_DIR="examples/scene_results/${DATA_DIR#*MyData/}/test_fraction_${TEST_FRACTION}/init_${INIT_TYPE}/common_ckpt_${COMMON_CKPT}/"

    echo "Data dir: ${DATA_DIR}"
    echo "Result dir: ${RESULT_PARENT_DIR}"


    # Train and Retrain

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/train_and_retrain/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --max_steps 10000 \
    --save_steps 5000 10000 \
    --eval_steps 5000 10000 \
    --fix_steps


    ## Refine and Retrain

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/novel_effect_1/refine_and_retrain/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${RESULT_PARENT_DIR}/train_and_retrain/ckpts/ckpt_4999_rank0.pt \
    --max_steps 10000 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --fix_steps 5000


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
    --render_only \
    --max_steps 10000 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --fix_steps 5000


    ## Refine (w/o reference) and Retrain

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/novel_effect_1/refine_wo_ref_and_retrain/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${RESULT_PARENT_DIR}/train_and_retrain/ckpts/ckpt_4999_rank0.pt \
    --refine_wo_ref \
    --max_steps 10000 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --fix_steps 5000

    echo "Finished scene: ${SCENE}"
    echo "----------------------------------"
done
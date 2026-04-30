SCENES=(
    bicycle
    bonsai
    counter
    garden
    kitchen
    room
    stump
)

DATA_FACTOR=4
TEST_FRACTION=0.9
INIT_TYPE=sfm
COMMON_CKPT_NAME=5k
COMMON_CKPT=4999
NOVEL_WEIGHT_DESC=novel_effect_1
FILTER_NAME=sfm_reproj

for SCENE in "${SCENES[@]}"
do
    echo "Running scene: ${SCENE}"

    DATA_DIR=../../MyData/Mip-Nerf-360/${SCENE}/
    RESULT_PARENT_DIR="examples/scene_results/${DATA_DIR#*MyData/}/test_fraction_${TEST_FRACTION}/init_${INIT_TYPE}/common_ckpt_${COMMON_CKPT_NAME}/"
    RENDERS_DIR=${RESULT_PARENT_DIR}/${NOVEL_WEIGHT_DESC}/refine_and_retrain/renders/
    CHECKPOINT_FILE=${RESULT_PARENT_DIR}/train_and_retrain/ckpts/ckpt_${COMMON_CKPT}_rank0.pt
    

    # # Train and Retrain

    # CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    # --data_dir ${DATA_DIR} \
    # --data_factor ${DATA_FACTOR} \
    # --result_dir ${RESULT_PARENT_DIR}/train_and_retrain/ \
    # --no-normalize-world-space \
    # --test_fraction ${TEST_FRACTION} \
    # --init_type ${INIT_TYPE} \
    # --max_steps 10000 \
    # --save_steps 5000 10000 \
    # --eval_steps 5000 10000 \
    # --fix_steps


    # ## Refine and Retrain

    # CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    # --data_dir ${DATA_DIR} \
    # --data_factor ${DATA_FACTOR} \
    # --result_dir ${RESULT_PARENT_DIR}/${NOVEL_WEIGHT_DESC}/refine_and_retrain/ \
    # --no-normalize-world-space \
    # --test_fraction ${TEST_FRACTION} \
    # --init_type ${INIT_TYPE} \
    # --ckpt ${CHECKPOINT_FILE} \
    # --max_steps 10000 \
    # --save_steps 10000 \
    # --eval_steps 10000 \
    # --fix_steps 5000

    
    ## Refine, Filter and Retrain

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/${NOVEL_WEIGHT_DESC}/refine_${FILTER_NAME}_filter_and_retrain/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${CHECKPOINT_FILE} \
    --novel_views_path ${RENDERS_DIR} \
    --filter_name ${FILTER_NAME} \
    --max_steps 10000 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --fix_steps 5000


    # ## Render and Retrain

    # CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    # --data_dir ${DATA_DIR} \
    # --data_factor ${DATA_FACTOR} \
    # --result_dir ${RESULT_PARENT_DIR}/${NOVEL_WEIGHT_DESC}/render_and_retrain/ \
    # --no-normalize-world-space \
    # --test_fraction ${TEST_FRACTION} \
    # --init_type ${INIT_TYPE} \
    # --ckpt ${CHECKPOINT_FILE} \
    # --novel_views_path ${RENDERS_DIR} \
    # --render_only \
    # --max_steps 10000 \
    # --save_steps 10000 \
    # --eval_steps 10000 \
    # --fix_steps 5000


    # ## Refine (w/o reference) and Retrain

    # CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    # --data_dir ${DATA_DIR} \
    # --data_factor ${DATA_FACTOR} \
    # --result_dir ${RESULT_PARENT_DIR}/${NOVEL_WEIGHT_DESC}/refine_wo_ref_and_retrain/ \
    # --no-normalize-world-space \
    # --test_fraction ${TEST_FRACTION} \
    # --init_type ${INIT_TYPE} \
    # --ckpt ${CHECKPOINT_FILE} \
    # --refine_wo_ref \
    # --max_steps 10000 \
    # --save_steps 10000 \
    # --eval_steps 10000 \
    # --fix_steps 5000

    echo "Finished scene: ${SCENE}"
    echo "----------------------------------"
done
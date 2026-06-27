SCENES=(
    # mono_sample_2018-04-17-V2-Log1
    # bicycle
    # bonsai
    # counter
    # garden
    # kitchen
    room
    # stump
)

DATA_FACTOR=4
TEST_FRACTION=0.9666666666666667
INIT_TYPE=sfm
NOVEL_LOSS_RATIO=1
NOVEL_SAMPLE_RATIO=-1
COMMON_CKPT_NAME=5k
COMMON_CKPT=9999
NOVEL_STRATEGY_FOLDER_NAME=novel_ratios_loss_${NOVEL_LOSS_RATIO}_sample_${NOVEL_SAMPLE_RATIO}
NOVEL_STRATEGY_FOLDER_NAME_FOR_RENDERS=$NOVEL_STRATEGY_FOLDER_NAME
FILTER_NAME=sfm_reproj

for SCENE in "${SCENES[@]}"
do
    echo "Running scene: ${SCENE}"
    
    DATA_DIR=../../MyData/Mip-Nerf-360/${SCENE}/
    RESULT_PARENT_DIR="examples/scene_results/${DATA_DIR#*MyData/}/test_fraction_${TEST_FRACTION}/init_${INIT_TYPE}/common_ckpt_${COMMON_CKPT_NAME}/"
    
    # DATA_DIR=../../MyData/ford-av-dataset/samples/${SCENE}/
    # RESULT_PARENT_DIR="examples/scene_results/ford-av-samples/${SCENE}/test_fraction_${TEST_FRACTION}/init_${INIT_TYPE}/common_ckpt_${COMMON_CKPT_NAME}/"
    
    RENDERS_DIR=${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME_FOR_RENDERS}/refine_and_retrain/renders/
    CHECKPOINT_FILE=${RESULT_PARENT_DIR}/train_and_retrain/ckpts/ckpt_${COMMON_CKPT}_rank0.pt
    

    # Train and Retrain
    printf "\n\n----------------------------------"
    printf "\nStarting Train and Retrain for: %s\n" "$SCENE"

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/train_and_retrain/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --max_steps 30000 \
    --save_steps 5000 10000 20000 30000 \
    --eval_steps 30000 \
    --fix_steps


    # # Refine and Retrain
    # printf "\n\n----------------------------------"
    # printf "\nStarting Refine and Retrain for: %s\n" "$SCENE"

    # CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    # --data_dir ${DATA_DIR} \
    # --data_factor ${DATA_FACTOR} \
    # --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/refine_and_retrain/ \
    # --no-normalize-world-space \
    # --test_fraction ${TEST_FRACTION} \
    # --init_type ${INIT_TYPE} \
    # --ckpt ${CHECKPOINT_FILE} \
    # --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    # --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    # --max_steps 10000 \
    # --save_steps 10000 \
    # --eval_steps 10000 \
    # --fix_steps 5000

    
    # # Render and Retrain
    # printf "\n\n----------------------------------"
    # printf "\nStarting Render and Retrain for: %s\n" "$SCENE"

    # CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    # --data_dir ${DATA_DIR} \
    # --data_factor ${DATA_FACTOR} \
    # --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/render_and_retrain/ \
    # --no-normalize-world-space \
    # --test_fraction ${TEST_FRACTION} \
    # --init_type ${INIT_TYPE} \
    # --ckpt ${CHECKPOINT_FILE} \
    # --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    # --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    # --novel_views_path ${RENDERS_DIR} \
    # --render_only \
    # --max_steps 10000 \
    # --save_steps 10000 \
    # --eval_steps 10000 \
    # --fix_steps 5000


    # # Refine, Filter and Retrain
    # printf "\n\n----------------------------------"
    # printf "\nStarting Refine, Filter and Retrain for: %s\n" "$SCENE"

    # CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    # --data_dir ${DATA_DIR} \
    # --data_factor ${DATA_FACTOR} \
    # --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/refine_${FILTER_NAME}_filter_and_retrain/ \
    # --no-normalize-world-space \
    # --test_fraction ${TEST_FRACTION} \
    # --init_type ${INIT_TYPE} \
    # --ckpt ${CHECKPOINT_FILE} \
    # --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    # --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    # --novel_views_path ${RENDERS_DIR} \
    # --filter_name ${FILTER_NAME} \
    # --max_steps 10000 \
    # --save_steps 10000 \
    # --eval_steps 10000 \
    # --fix_steps 5000



    # # Refine (w/o reference) and Retrain
    # printf "\n\n----------------------------------"
    # printf "\nStarting Refine (w/o reference) and Retrain for: %s\n" "$SCENE"

    # CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    # --data_dir ${DATA_DIR} \
    # --data_factor ${DATA_FACTOR} \
    # --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/refine_wo_ref_and_retrain/ \
    # --no-normalize-world-space \
    # --test_fraction ${TEST_FRACTION} \
    # --init_type ${INIT_TYPE} \
    # --ckpt ${CHECKPOINT_FILE} \
    # --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    # --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    # --refine_wo_ref \
    # --max_steps 10000 \
    # --save_steps 10000 \
    # --eval_steps 10000 \
    # --fix_steps 5000

    echo "----------------------------------"
    echo "Finished scene: ${SCENE}"
    echo "----------------------------------"
done
SCENES=(
    # bicycle
    # bonsai
    # counter
    # garden
    # kitchen
    # room
    # stump
)

DATA_FACTOR=4
TEST_FRACTION=0.9
INIT_TYPE=sfm
NOVEL_LOSS_RATIO=1
COMMON_CKPT_NAME=30k
COMMON_CKPT=29999
FILTER_NAME=dino_classifier

for SCENE in "${SCENES[@]}"
do
    echo "Running scene: ${SCENE}"
    
    DATA_DIR=../../MyData/Mip-Nerf-360/${SCENE}/
    RESULT_PARENT_DIR="examples/scene_results/${DATA_DIR#*MyData/}/test_fraction_${TEST_FRACTION}/init_${INIT_TYPE}/common_ckpt_${COMMON_CKPT_NAME}/"   
    
    PREV_RESULT_PARENT_DIR="examples/scene_results/${DATA_DIR#*MyData/}/test_fraction_${TEST_FRACTION}/init_${INIT_TYPE}/common_ckpt_5k/"   
    CHECKPOINT_FILE=${PREV_RESULT_PARENT_DIR}/train_and_retrain/ckpts/ckpt_${COMMON_CKPT}_rank0.pt

    CLASSIFIER_OUTPUT_CSV_FILE="examples/scene_results/Mip-Nerf-360/eval_results_indivdual_classifier/30k_refined_scene_specific_results_v2/${SCENE}_predictions.csv"

    #====Base====================================================================
    
    # # GS_0 (Eval @ 30k)
    # printf "\n\n----------------------------------"
    # printf "\nGS_0 (Eval @ 30k) for: %s\n" "$SCENE"

    # CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    # --data_dir ${DATA_DIR} \
    # --data_factor ${DATA_FACTOR} \
    # --result_dir ${RESULT_PARENT_DIR}/GS_0/ \
    # --no-normalize-world-space \
    # --test_fraction ${TEST_FRACTION} \
    # --init_type ${INIT_TYPE} \
    # --ckpt ${CHECKPOINT_FILE} \
    # --max_steps 30000 \
    # --save_steps 30000\
    # --eval_steps 30000 \
    # --fix_steps

    # # GS_10 (Train and Eval @ 40k)
    # printf "\n\n----------------------------------"
    # printf "\nGS_10 (Eval @ 40k) for: %s\n" "$SCENE"

    # CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    # --data_dir ${DATA_DIR} \
    # --data_factor ${DATA_FACTOR} \
    # --result_dir ${RESULT_PARENT_DIR}/GS_10/ \
    # --no-normalize-world-space \
    # --test_fraction ${TEST_FRACTION} \
    # --init_type ${INIT_TYPE} \
    # --ckpt ${CHECKPOINT_FILE} \
    # --max_steps 40000 \
    # --save_steps 40000 \
    # --eval_steps 40000 \
    # --fix_steps

    # ====Fix and Save Images=====================================================
    
    # # GS_0f (Refine and Save Images @ 30k)
    # printf "\n\n----------------------------------"
    # printf "\nGS_0f (Refine and Save Images @ 30k) for: %s\n" "$SCENE"

    # CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    # --data_dir ${DATA_DIR} \
    # --data_factor ${DATA_FACTOR} \
    # --result_dir ${RESULT_PARENT_DIR}/GS_0f/ \
    # --no-normalize-world-space \
    # --test_fraction ${TEST_FRACTION} \
    # --init_type ${INIT_TYPE} \
    # --ckpt ${CHECKPOINT_FILE} \
    # --max_steps 30000 \
    # --save_steps \
    # --eval_steps \
    # --fix_steps 30000

    #====NOVEL SAMPLE RATIO = 1=================================================

    NOVEL_SAMPLE_RATIO=1 # use only novel views for training
    NOVEL_STRATEGY_FOLDER_NAME=novel_ratios_loss_${NOVEL_LOSS_RATIO}_sample_${NOVEL_SAMPLE_RATIO}_no_alpha_mask

    #====Run Difix without Classifier filter=====================================

    # # GS_10f (Difix without Classifier filter, Trained for 10k steps after 30k)
    # printf "\n\n----------------------------------"
    # printf "\nGS_10f (Difix without Classifier filter, Trained for 10k steps after 30k) for: %s\n" "$SCENE"
    # CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    # --data_dir ${DATA_DIR} \
    # --data_factor ${DATA_FACTOR} \
    # --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_10f/ \
    # --no-normalize-world-space \
    # --test_fraction ${TEST_FRACTION} \
    # --init_type ${INIT_TYPE} \
    # --ckpt ${CHECKPOINT_FILE} \
    # --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    # --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    # --novel_views_path ${RESULT_PARENT_DIR}/GS_0f/renders/novel/${COMMON_CKPT}/ \
    # --classifier_output_csv_file ${CLASSIFIER_OUTPUT_CSV_FILE} \
    # --max_steps 40000 \
    # --save_steps 40000 \
    # --eval_steps 40000 \
    # --fix_steps 30000


    #====Run Difix with Classifier filter========================================
    
    # GS_10fc (Difix with Classifier filter, Trained for 10k steps after 30k)
    printf "\n\n----------------------------------"
    printf "\nGS_10fc (Difix with Classifier filter, Trained for 10k steps after 30k) for: %s\n" "$SCENE"
    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_10fc/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${CHECKPOINT_FILE} \
    --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    --novel_views_path ${RESULT_PARENT_DIR}/GS_0f/renders/novel/${COMMON_CKPT}/ \
    --classifier_output_csv_file ${CLASSIFIER_OUTPUT_CSV_FILE} \
    --filter_name ${FILTER_NAME} \
    --max_steps 40000 \
    --save_steps 40000 \
    --eval_steps 40000 \
    --fix_steps 30000

    # #====NOVEL SAMPLE RATIO = -1========================================

    # NOVEL_SAMPLE_RATIO=-1 # use only novel and original views in the ratio of their counts
    # NOVEL_STRATEGY_FOLDER_NAME=novel_ratios_loss_${NOVEL_LOSS_RATIO}_sample_${NOVEL_SAMPLE_RATIO}_no_alpha_mask

    # #====Run Difix without Classifier filter=====================================

    # # GS_10f (Difix without Classifier filter, Trained for 10k steps after 30k)
    # printf "\n\n----------------------------------"
    # printf "\nGS_10f (Difix without Classifier filter, Trained for 10k steps after 30k) for: %s\n" "$SCENE"
    # CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    # --data_dir ${DATA_DIR} \
    # --data_factor ${DATA_FACTOR} \
    # --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_10f/ \
    # --no-normalize-world-space \
    # --test_fraction ${TEST_FRACTION} \
    # --init_type ${INIT_TYPE} \
    # --ckpt ${CHECKPOINT_FILE} \
    # --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    # --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    # --novel_views_path ${RESULT_PARENT_DIR}/GS_0f/renders/novel/${COMMON_CKPT}/ \
    # --classifier_output_csv_file ${CLASSIFIER_OUTPUT_CSV_FILE} \
    # --max_steps 40000 \
    # --save_steps 40000 \
    # --eval_steps 40000 \
    # --fix_steps 30000


    # #====Run Difix with Classifier filter========================================
    
    # # GS_10fc (Difix with Classifier filter, Trained for 10k steps after 30k)
    # printf "\n\n----------------------------------"
    # printf "\nGS_10fc (Difix with Classifier filter, Trained for 10k steps after 30k) for: %s\n" "$SCENE"
    # CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    # --data_dir ${DATA_DIR} \
    # --data_factor ${DATA_FACTOR} \
    # --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_10fc/ \
    # --no-normalize-world-space \
    # --test_fraction ${TEST_FRACTION} \
    # --init_type ${INIT_TYPE} \
    # --ckpt ${CHECKPOINT_FILE} \
    # --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    # --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    # --novel_views_path ${RESULT_PARENT_DIR}/GS_0f/renders/novel/${COMMON_CKPT}/ \
    # --classifier_output_csv_file ${CLASSIFIER_OUTPUT_CSV_FILE} \
    # --filter_name ${FILTER_NAME} \
    # --max_steps 40000 \
    # --save_steps 40000 \
    # --eval_steps 40000 \
    # --fix_steps 30000





    echo "----------------------------------"
    echo "Finished scene: ${SCENE}"
    echo "----------------------------------"
done
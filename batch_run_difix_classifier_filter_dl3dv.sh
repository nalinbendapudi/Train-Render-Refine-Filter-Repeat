SCENES=(
    "04b9beb07e5d9d48198b877b2d0342e92a754d200c8eef3bacc3e2c7f94747e1"
    # "5deb261c38bc5dc270eb084bcc2f8018269966efe9640192538de976e4eb67eb"
    # "010cf78dd4aa5f9de2c83cb366e4ac7fd44791eee7dc5234b3104f752f76b15b"
    # "27daa60e1fa93a4f7dd57a2ce5efcaff9c6937ed8c144c824f3ad39aae5ed95d"
    # "2036ef5406a4d39e23e7fbf19156f67112bb57c529611476a6835d3ce88148c9"
    # "ad836db69995dbb8f88bd86c1d8bcf16f3ec818b5aad762af061b90c1a1d4ffe"
    # "eb4dd9619edef0f2ee62ae7ce80d8f2f6862ca4d6e8ce087741cd86d86e955a0"
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
    
    DATA_DIR=../../MyData/DL3DV/outdoor_scenes/${SCENE}/
    RESULT_PARENT_DIR="examples/scene_results/${DATA_DIR#*MyData/}/test_fraction_${TEST_FRACTION}/init_${INIT_TYPE}/common_ckpt_${COMMON_CKPT_NAME}/"   
    
    CLASSIFIER_OUTPUT_CSV_FILE=""

    #====Base====================================================================
    
    # GS_0 (Eval @ 30k)
    printf "\n\n----------------------------------"
    printf "\nGS_0 (Train andEval @ 30k) for: %s\n" "$SCENE"

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/GS_0/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --max_steps 30000 \
    --save_steps 30000\
    --eval_steps 30000 \
    --fix_steps

    # CHECKPOINT_FILE=${RESULT_PARENT_DIR}/GS_0/ckpts/ckpt_${COMMON_CKPT}_rank0.pt

    # # GS_10 (Train and Eval @ 40k)
    # printf "\n\n----------------------------------"
    # printf "\nGS_10 (Train and Eval @ 40k) for: %s\n" "$SCENE"

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

    # NOVEL_SAMPLE_RATIO=1 # use only novel views for training
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
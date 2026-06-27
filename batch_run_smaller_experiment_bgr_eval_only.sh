SCENES=(
    # bicycle
    # bonsai
    counter
    garden
    kitchen
    # room
    stump
)

DATA_FACTOR=4
# TEST_FRACTION=0.9666666666666667
TEST_FRACTION=0.9
INIT_TYPE=sfm
NOVEL_LOSS_RATIO=1
NOVEL_SAMPLE_RATIO=1 # use only novel views for training
COMMON_CKPT_NAME=30k
COMMON_CKPT=29999
NOVEL_STRATEGY_FOLDER_NAME=novel_ratios_loss_${NOVEL_LOSS_RATIO}_sample_${NOVEL_SAMPLE_RATIO}_no_alpha_mask
# FILTER_NAME=sfm_reproj

for SCENE in "${SCENES[@]}"
do
    echo "Running scene: ${SCENE}"
    
    DATA_DIR=../../MyData/Mip-Nerf-360/${SCENE}/
    RESULT_PARENT_DIR="examples/scene_results/${DATA_DIR#*MyData/}/test_fraction_${TEST_FRACTION}/init_${INIT_TYPE}/common_ckpt_${COMMON_CKPT_NAME}/"   
    
    PREV_RESULT_PARENT_DIR="examples/scene_results/${DATA_DIR#*MyData/}/test_fraction_${TEST_FRACTION}/init_${INIT_TYPE}/common_ckpt_5k/"   
    CHECKPOINT_FILE=${PREV_RESULT_PARENT_DIR}/train_and_retrain/ckpts/ckpt_${COMMON_CKPT}_rank0.pt
    SUBSET_INDICES_TXT_FILE=${PREV_RESULT_PARENT_DIR}/train_and_retrain/renders/val/${COMMON_CKPT}/subset_indices.txt
    GT_VIEWS_PATH=${PREV_RESULT_PARENT_DIR}/train_and_retrain/renders/val/${COMMON_CKPT}/GT/

    #====Base================================================================
    
    # GS_br_0 (Eval @ 30k)
    printf "\n\n----------------------------------"
    printf "\nGS_br_0 (Eval @ 30k) for: %s\n" "$SCENE"

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/GS_0/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${RESULT_PARENT_DIR}/GS_0/ckpts/ckpt_29999_rank0.pt \
    --subset_indices_txt_file ${SUBSET_INDICES_TXT_FILE} \
    --use_bad_renders_for_eval  \
    --max_steps 30000 \
    --save_steps \
    --eval_steps 30000 \
    --fix_steps


    # GS_gr_0 (Eval @ 30k)
    printf "\n\n----------------------------------"
    printf "\nGS_gr_0 (Eval @ 30k) for: %s\n" "$SCENE"

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/GS_0/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${RESULT_PARENT_DIR}/GS_0/ckpts/ckpt_29999_rank0.pt \
    --subset_indices_txt_file ${SUBSET_INDICES_TXT_FILE} \
    --use_good_renders_for_eval  \
    --max_steps 30000 \
    --save_steps \
    --eval_steps 30000 \
    --fix_steps


    # GS_bgr_0 (Eval @ 30k)
    printf "\n\n----------------------------------"
    printf "\nGS_bgr_0 (Eval @ 30k) for: %s\n" "$SCENE"

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/GS_0/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${RESULT_PARENT_DIR}/GS_0/ckpts/ckpt_29999_rank0.pt \
    --subset_indices_txt_file ${SUBSET_INDICES_TXT_FILE} \
    --use_bad_renders_for_eval  \
    --use_good_renders_for_eval  \
    --max_steps 30000 \
    --save_steps \
    --eval_steps 30000 \
    --fix_steps


    #====Fixed================================================================

    # GS_br_f (Train further with fixed versions of bad renders from 30k to 40k)
    printf "\n\n----------------------------------"
    printf "\nGS_br_f (Train further with fixed versions of bad renders) for: %s\n" "$SCENE"

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_br_f/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_br_f/ckpts/ckpt_39999_rank0.pt \
    --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    --subset_indices_txt_file ${SUBSET_INDICES_TXT_FILE} \
    --use_bad_renders_for_retrain \
    --use_bad_renders_for_eval  \
    --max_steps 40000 \
    --save_steps  \
    --eval_steps 40000 \
    --fix_steps 


    # GS_gr_f (Train further with fixed versions of good renders from 30k to 40k)
    printf "\n\n----------------------------------"
    printf "\nGS_gr_f (Train further with fixed versions of good renders) for: %s\n" "$SCENE"

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_gr_f/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_gr_f/ckpts/ckpt_39999_rank0.pt \
    --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    --subset_indices_txt_file ${SUBSET_INDICES_TXT_FILE} \
    --use_good_renders_for_retrain \
    --use_good_renders_for_eval  \
    --max_steps 40000 \
    --save_steps  \
    --eval_steps 40000 \
    --fix_steps 


    # GS_bgr_f (Train further with fixed versions of bad and good renders from 30k to 40k)
    printf "\n\n----------------------------------"
    printf "\nGS_bgr_f (Train further with fixed versions of bad and good renders) for: %s\n" "$SCENE"

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_bgr_f/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_bgr_f/ckpts/ckpt_39999_rank0.pt \
    --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    --subset_indices_txt_file ${SUBSET_INDICES_TXT_FILE} \
    --use_bad_renders_for_retrain \
    --use_bad_renders_for_eval  \
    --use_good_renders_for_retrain \
    --use_good_renders_for_eval  \
    --max_steps 40000 \
    --save_steps  \
    --eval_steps 40000 \
    --fix_steps 


    #====Fixed w/o ref================================================================

    # GS_br_fworef (Train further with fixed (w/o ref) versions of bad renders from 30k to 40k)
    printf "\n\n----------------------------------"
    printf "\nGS_br_fworef (Train further with fixed (w/o ref) versions of bad renders) for: %s\n" "$SCENE"

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_br_fworef/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_br_fworef/ckpts/ckpt_39999_rank0.pt \
    --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    --subset_indices_txt_file ${SUBSET_INDICES_TXT_FILE} \
    --use_bad_renders_for_retrain \
    --use_bad_renders_for_eval  \
    --max_steps 40000 \
    --save_steps  \
    --eval_steps 40000 \
    --fix_steps 


    # GS_gr_fworef (Train further with fixed (w/o ref) versions of good renders from 30k to 40k)
    printf "\n\n----------------------------------"
    printf "\nGS_gr_fworef (Train further with fixed (w/o ref) versions of good renders) for: %s\n" "$SCENE"

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_gr_fworef/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_gr_fworef/ckpts/ckpt_39999_rank0.pt \
    --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    --subset_indices_txt_file ${SUBSET_INDICES_TXT_FILE} \
    --use_good_renders_for_retrain \
    --use_good_renders_for_eval  \
    --max_steps 40000 \
    --save_steps  \
    --eval_steps 40000 \
    --fix_steps 


    # GS_bgr_fworef (Train further with fixed (w/o ref) versions of bad and good renders from 30k to 40k)
    printf "\n\n----------------------------------"
    printf "\nGS_bgr_fworef (Train further with fixed (w/o ref) versions of bad and good renders) for: %s\n" "$SCENE"

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_bgr_fworef/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_bgr_fworef/ckpts/ckpt_39999_rank0.pt \
    --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    --subset_indices_txt_file ${SUBSET_INDICES_TXT_FILE} \
    --use_bad_renders_for_retrain \
    --use_bad_renders_for_eval  \
    --use_good_renders_for_retrain \
    --use_good_renders_for_eval  \
    --max_steps 40000 \
    --save_steps  \
    --eval_steps 40000 \
    --fix_steps 


    # ===Original================================================================

    # GS_br_o (Train further with orignals of bad renders from 30k to 40k)
    printf "\n\n----------------------------------"
    printf "\nGS_br_o (Train further with orignals of bad renders) for: %s\n" "$SCENE"

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_br_o/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_br_o/ckpts/ckpt_39999_rank0.pt \
    --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    --render_only \
    --subset_indices_txt_file ${SUBSET_INDICES_TXT_FILE} \
    --use_bad_renders_for_retrain \
    --use_bad_renders_for_eval  \
    --use_originals_instead_of_renders \
    --gt_views_path ${GT_VIEWS_PATH} \
    --max_steps 40000 \
    --save_steps  \
    --eval_steps 40000 \
    --fix_steps 


    # GS_gr_o (Train further with orignals of good renders from 30k to 40k)
    printf "\n\n----------------------------------"
    printf "\nGS_gr_o (Train further with orignals of good renders) for: %s\n" "$SCENE"

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_gr_o/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_gr_o/ckpts/ckpt_39999_rank0.pt \
    --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    --render_only \
    --subset_indices_txt_file ${SUBSET_INDICES_TXT_FILE} \
    --use_good_renders_for_retrain \
    --use_good_renders_for_eval  \
    --use_originals_instead_of_renders \
    --gt_views_path ${GT_VIEWS_PATH} \
    --max_steps 40000 \
    --save_steps  \
    --eval_steps 40000 \
    --fix_steps 

    # GS_bgr_o (Train further with orignals of bad anf good renders from 30k to 40k)
    printf "\n\n----------------------------------"
    printf "\nGS_bgr_o (Train further with orignals of bad and good renders) for: %s\n" "$SCENE"

    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_bgr_o/ \
    --no-normalize-world-space \
    --test_fraction ${TEST_FRACTION} \
    --init_type ${INIT_TYPE} \
    --ckpt ${RESULT_PARENT_DIR}/${NOVEL_STRATEGY_FOLDER_NAME}/GS_bgr_o/ckpts/ckpt_39999_rank0.pt \
    --novel_loss_ratio ${NOVEL_LOSS_RATIO} \
    --novel_sample_ratio ${NOVEL_SAMPLE_RATIO} \
    --render_only \
    --subset_indices_txt_file ${SUBSET_INDICES_TXT_FILE} \
    --use_bad_renders_for_retrain \
    --use_bad_renders_for_eval  \
    --use_good_renders_for_retrain \
    --use_good_renders_for_eval  \
    --use_originals_instead_of_renders \
    --gt_views_path ${GT_VIEWS_PATH} \
    --max_steps 40000 \
    --save_steps  \
    --eval_steps 40000 \
    --fix_steps 


    echo "----------------------------------"
    echo "Finished scene: ${SCENE}"
    echo "----------------------------------"
done

export CUDA_VISIBLE_DEVICES=2,5,6,7
export CUDA_LAUNCH_BLOCKING=1

torchrun --nproc_per_node=4 fine_tune.py \
        --model_name_or_path /group/ossmodelzoo/sequence_learning/weights/nlp-pretrained-model/meta-llama/Llama-2-7b-hf \
        --bf16 False \
        --output_dir new_outputs/diff_outputs   \
        --model_max_length 4096 \
        --use_flash_attn False \
        --use_full_attn True \
        --low_rank_training True \
        --num_train_epochs 1  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 1     \
        --gradient_accumulation_steps 8     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 500     \
        --save_total_limit 2     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 1     \
        --deepspeed "ds_configs/stage2.json" \
        --tf32 False \
        --max_steps 2000  \
        --layer_diff True  \
        --head_diff True  \
        --resume_from_checkpoint False


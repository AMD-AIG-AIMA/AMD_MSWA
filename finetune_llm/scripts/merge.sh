
checkpoint=new_outputs/normal_outputs/checkpoint-2000/
save_path=hf_outputs/llama2_7b/normal
#save_path=hf_outputs/llama2_7b/2_2_4_8/diff_1024

cd ${checkpoint} && python zero_to_fp32.py . pytorch_model.bin && cd ../../..

python get_trainable_weights.py --checkpoint_path ${checkpoint} --trainable_params "embed,norm"

python merge_lora_weights_and_save_hf_model.py \
        --peft_model ${checkpoint} \
        --save_path ${save_path}
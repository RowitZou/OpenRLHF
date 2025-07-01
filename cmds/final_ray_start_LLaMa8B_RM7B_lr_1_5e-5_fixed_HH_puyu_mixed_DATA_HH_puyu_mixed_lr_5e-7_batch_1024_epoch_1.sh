train_batch_size=1024
rollout_batch_size=1024
lr=5e-7
epoch=1
data_type=Mixed-1252K
policy_model_name=LLaMa3-8B-Instruct
# policy_model_name=InternLM3-8B-Instruct-wo-RL
rm_name=RM_SFT_reward_pt_7b_final_DATA_HH_puyu_mixed_Node_2_LR_1_5e-5_STEP_905_hf_fixed

# actor_pretrain_path=/cpfs01/shared/alillm_hs/zouyicheng/rm_pretrain/ckpts_val/official_Kepler_dense_8B_20241225k_256k_enhance_256k_3_1_20500_FT_internlm3_32k_s1_finalrc11_256gpu_4073_s2_128k_internlm3_s2_final_rc19_20250108d_845_hf
actor_pretrain_path=/cpfs01/shared/alillm_hs/zouyicheng/rm_pretrain/all_models/Meta-Llama-3.1-8B-Instruct
# /cpfs01/shared/alillm_hs/liushichun1/models/internlm3-8b-instruct-wo-rl
reward_pretrain_path=/cpfs01/shared/public/zouyicheng/RM_SFT_reward_pt_7b_final_DATA_HH_puyu_mixed_Node_2_LR_1_5e-5_STEP_905_hf
reward_remote_url=10.130.0.51:30000
prompt_data_path=/cpfs01/shared/alillm_hs/zouyicheng/rm_pretrain/data/ppo/mixed/train
total_sample_num=1252429

name="final-ppo-ray-policy_${policy_model_name}-${rm_name}_data_${data_type}_bsz_${train_batch_size}_lr_${lr}_epoch_${epoch}"

save_steps=$(( (total_sample_num / rollout_batch_size) / 15 ))

cd /cpfs01/shared/alillm_hs/zouyicheng/OpenRLHF
TARGET_FILE="/cpfs01/shared/alillm_hs/zouyicheng/OpenRLHF/addr/addr_${name}.txt"
RANK=${RANK:-0}
MASTER_PORT=6379
MASTER_ADDR=${MASTER_ADDR}
echo "MASTER_ADDR: $MASTER_ADDR"



echo "Rank $RANK is running on $MASTER_ADDR"
if [ "$RANK" -eq 0 ]; then 
    echo "Starting head node (RANK=${RANK}) on port $MASTER_PORT..."
    
    MASTER_ADDR=${MASTER_ADDR}
    echo "$MASTER_ADDR" > "$TARGET_FILE"

    ray start --head --num-gpus 8 --block &
    sleep 60
    
    echo "Executing main program on head node..."
    # TODO:     # --colocate_critic_reward \
    ray job submit --address="http://127.0.0.1:8265"  \
    --runtime-env-json='{"working_dir": "/cpfs01/shared/alillm_hs/zouyicheng/OpenRLHF"}' \
    -- python -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 0 \
    --ref_num_gpus_per_node 8 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 8 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 16 \
    --vllm_tensor_parallel_size 1 \
    --vllm_sync_backend nccl \
    --colocate_actor_ref \
    --pretrain $actor_pretrain_path \
    --remote_rm_url $reward_remote_url \
    --reward_pretrain $reward_pretrain_path \
    --save_path /cpfs01/shared/alillm_hs/zouyicheng/OpenRLHF/ckpts/${name} \
    --ckpt_path /cpfs01/shared/alillm_hs/zouyicheng/OpenRLHF/ckpts/${name} \
    --micro_train_batch_size 2 \
    --train_batch_size $train_batch_size \
    --micro_rollout_batch_size 32 \
    --rollout_batch_size $rollout_batch_size \
    --num_episodes $epoch \
    --prompt_max_len 4096 \
    --generate_max_len 4096 \
    --save_steps $save_steps \
    --max_ckpt_num $save_steps \
    --save_hf_ckpt \
    --zero_stage 1 \
    --bf16 \
    --lambd 1 \
    --actor_learning_rate $lr \
    --critic_learning_rate 1e-5 \
    --actor_min_learning_rate $lr \
    --critic_min_learning_rate 1e-6 \
    --lr_warmup_ratio 0.03 \
    --critic_pretrain $actor_pretrain_path \
    --init_kl_coef 0 \
    --prompt_data json@${prompt_data_path} \
    --input_key message_data \
    --label_key ref_message_data \
    --ref_mode \
    --reward_mean -5.0 \
    --reward_std 10.0 \
    --normalize_reward \
    --packing_samples \
    --overlap_comm \
    --flash_attn \
    --gradient_checkpointing \
    --apply_chat_template \
    --load_checkpoint \
    --use_tensorboard /cpfs01/shared/alillm_hs/zouyicheng/OpenRLHF/logs/${name}
    
else 
    sleep 30
    MASTER_ADDR=$(cat "$TARGET_FILE")

    echo "Starting worker node (RANK=${RANK}), connecting to ${MASTER_ADDR}:${MASTER_PORT}..."
    ray start --address ${MASTER_ADDR}:${MASTER_PORT}  --num-gpus 8 --block &
    
    sleep 120
    # worker保持运行状态
    while true; do
        # 获取ray status的输出
        status=$(ray status 2>&1)

        # 检查是否有 active 的 node
        if echo "$status" | grep -q "Active:"; then
            # 如果有 active 的 node，继续 sleep
            echo "Active nodes found. Sleeping for 10 min..."
            sleep 600
        else
            # 如果没有 active 的 node，退出脚本
            echo "No active nodes found. Exiting..."
            exit 0
        fi
    done

fi

#!/bin/bash

#########################
# Configurable Variables
#########################

export HOME="/cpfs01/shared/alillm_hs/yangyuming/"

# === Training parameters ===
train_batch_size=1024
rollout_batch_size=1024
lr=1e-6
epoch=1
data_type="Mixed_1252K"

# === Model names ===
policy_model_name="InternLM3-8B-Instruct-wo-RL"
rm_name="RM_Skyworks_LLama_31_8B"

# === Paths ===
actor_pretrain_path="/cpfs01/shared/alillm_hs/liushichun1/models/internlm3-8b-instruct-wo-rl"
reward_pretrain_path="/cpfs01/shared/alillm_hs/liushichun1/models/Skywork-Reward-Llama-3.1-8B"
reward_remote_url="10.130.1.105:30000"
prompt_data_path="/cpfs01/shared/alillm_hs/zouyicheng/rm_pretrain/data/ppo/mixed/train/"
total_sample_num=1252429

# === Ray & DLC settings ===
dlc_workspace_id="wssp4wsoiyvfp6ww"
dlc_data_sources="data1ewbw1ztmmyh,data1bgvj0n14to0,data1dfp0cngxv41,data1ubhj4714msc,data1xj7ojru0t4t,data4n4f7sfaxa5g,data1o8qdjce0kd0"
docker_image="pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/pjlab-eflops/lingjun-pytorch-training:2.3-24.03-gu8-gpu-py310-cu124-ubuntu22.04"

# === Derived settings ===
name="ppo-ray-policy_${policy_model_name}-${rm_name}_data_${data_type}_bsz_${train_batch_size}_lr_${lr}_epoch_${epoch}"
save_steps=$(( (total_sample_num / rollout_batch_size) / 15 ))
ray_script_path="/cpfs01/shared/alillm_hs/yangyuming/OpenRLHF"
target_file="${ray_script_path}/addr/addr_${name}.txt"

# === Set rank and master address ===
RANK=${RANK:-0}
MASTER_PORT=6379

#########################
# DLC Submission Logic
#########################

cur_script_path=$(realpath "$0")
# Only submit DLC job if not running inside DLC and rank is 0
if [ -z "$RUN_FROM_DLC" ] && [ "$RANK" -eq 0 ]; then
    echo "⏳ Submitting DLC job: $name"

    cmd="sudo su && . /cpfs01/shared/alillm_hs/zouyicheng/.bashrc && cd $ray_script_path && conda activate rlhf && \
export RUN_FROM_DLC=1 && bash $cur_script_path"

    /cpfs01/shared/public/dlc create job --config /cpfs01/shared/alillm_hs/yangyuming/dlc.config \
        --name "$name" \
        --worker_count 4 \
        --kind PyTorchJob \
        --worker_cpu 60 \
        --worker_gpu 8 \
        --worker_memory 1280Gi \
        --worker_shared_memory 256Gi \
        --workspace_id "$dlc_workspace_id" \
        --data_sources "$dlc_data_sources" \
        --worker_image "$docker_image" \
        --command "$cmd" \
        --priority 4

    exit 0
fi

#########################
# PPO-Ray Execution Logic
#########################

cd "$ray_script_path"
MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
echo "RANK: $RANK, MASTER_ADDR: $MASTER_ADDR"

if [ "$RANK" -eq 0 ]; then
    echo "🌟 Starting Ray head node..."
    mkdir -p "$(dirname "$target_file")"
    echo "$MASTER_ADDR" > "$target_file"

    ray start --head --num-gpus 8 --block &
    sleep 120

    echo "🚀 Launching PPO training job..."
    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json="{\"working_dir\": \"$ray_script_path\"}" \
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
            --pretrain "$actor_pretrain_path" \
            --remote_rm_url "$reward_remote_url" \
            --reward_pretrain "$reward_pretrain_path" \
            --max_rm_model_length 4096 \
            --max_rm_response_length 2048 \
            --save_path "$ray_script_path/ckpts/${name}" \
            --ckpt_path "$ray_script_path/ckpts/${name}" \
            --micro_train_batch_size 2 \
            --train_batch_size "$train_batch_size" \
            --micro_rollout_batch_size 32 \
            --rollout_batch_size "$rollout_batch_size" \
            --num_episodes "$epoch" \
            --prompt_max_len 4096 \
            --generate_max_len 4096 \
            --save_steps "$save_steps" \
            --max_ckpt_num "$save_steps" \
            --save_hf_ckpt \
            --zero_stage 1 \
            --bf16 \
            --lambd 1 \
            --actor_learning_rate "$lr" \
            --critic_learning_rate 1e-5 \
            --actor_min_learning_rate "$lr" \
            --critic_min_learning_rate 1e-6 \
            --lr_warmup_ratio 0.03 \
            --critic_pretrain "$actor_pretrain_path" \
            --init_kl_coef 0 \
            --prompt_data "json@${prompt_data_path}" \
            --input_key message_data \
            --label_key ref_message_data \
            --normalize_reward \
            --packing_samples \
            --overlap_comm \
            --flash_attn \
            --gradient_checkpointing \
            --apply_chat_template \
            --load_checkpoint \
            --use_tensorboard "$ray_script_path/logs/${name}"
else
    sleep 120
    MASTER_ADDR=$(cat "$target_file")

    echo "🛠️ Starting Ray worker node and connecting to $MASTER_ADDR:$MASTER_PORT..."
    ray start --address "$MASTER_ADDR:$MASTER_PORT" --num-gpus 8 --block &

    sleep 120
    while true; do
        status=$(ray status 2>&1)
        if echo "$status" | grep -q "Active:"; then
            echo "🕒 Active Ray nodes found. Sleeping for 10 min..."
            sleep 600
        else
            echo "✅ Ray cluster finished. Exiting worker..."
            exit 0
        fi
    done
fi

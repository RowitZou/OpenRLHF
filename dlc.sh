#!/bin/bash
# hs: ws1lu4iyv5yjjyvp
# ddd: ws1ujefpjyfgqjwp

# zouyicheng data source data1ewbw1ztmmyh
# doushihan data source data1dfp0cngxv41
# liushichun1 data source data1ubhj4714msc

export HOME="/cpfs01/shared/llm_ddd/zouyicheng/"


function commit {
    num_nodes=4
    name="ppo-ray-policy_${policy_model_name}-${rm_name}_data_${data_type}_bsz_${train_batch_size}_lr_${lr}_epoch_${epoch}"
    num_tasks_per_node=1
    node_cpus=60
    num_gpus=8
    node_mems=1280Gi


    cmd="sudo su && . /cpfs01/shared/llm_ddd/zouyicheng/.bashrc && cd /cpfs01/shared/llm_ddd/zouyicheng/OpenRLHF && conda activate rlhf && \
bash /cpfs01/shared/llm_ddd/zouyicheng/OpenRLHF/cmds/final_ray_start_Internlm8B_RM1.8B_puyu_DATA_puyu_lr_5e-7_batch_128_epoch_4.sh"

    /cpfs01/shared/public/dlc create job --config /cpfs01/shared/llm_ddd/zouyicheng/dlc.config \
    --name $name \
    --worker_count $num_nodes \
    --kind PyTorchJob \
    --worker_cpu $node_cpus \
    --worker_gpu $num_gpus \
    --worker_memory $node_mems \
    --workspace_id ws1ujefpjyfgqjwp \
    --data_sources data1ewbw1ztmmyh,data1bgvj0n14to0,data1dfp0cngxv41,data1ubhj4714msc,data1xj7ojru0t4t  \
    --worker_image pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/pjlab-eflops/lingjun-pytorch-training:2.3-24.03-gu8-gpu-py310-cu124-ubuntu22.04 \
    --worker_shared_memory 256Gi \
    --command "$cmd" \
    --priority 4
}

train_batch_size=128
lr=5e-7
epoch=4
# data_type=HH-905K
data_type=Bench-347K
# policy_model_name=Qwen2.5-7B-Instruct
policy_model_name=InternLM8B
rm_name=final_RM1.8B_puyu

commit 
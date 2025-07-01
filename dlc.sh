#!/bin/bash
# hs: ws1lu4iyv5yjjyvp
# ddd: ws1ujefpjyfgqjwp
# ddd_new: wssp4wsoiyvfp6ww

# zouyicheng data source data1ewbw1ztmmyh
# doushihan data source data1dfp0cngxv41
# liushichun1 data source data1ubhj4714msc

export HOME="/cpfs01/shared/alillm_hs/zouyicheng/"


function commit {
    num_nodes=12
    name="ppo-ray-policy_${policy_model_name}-${rm_name}_data_${data_type}_bsz_${train_batch_size}_lr_${lr}_epoch_${epoch}"
    num_tasks_per_node=1
    node_cpus=60
    num_gpus=8
    node_mems=1280Gi


    cmd="sudo su && . /cpfs01/shared/alillm_hs/zouyicheng/.bashrc && cd /cpfs01/shared/alillm_hs/zouyicheng/OpenRLHF && conda activate rlhf && \
bash /cpfs01/shared/alillm_hs/zouyicheng/OpenRLHF/cmds/final_ray_start_Qwen32B_RM7B_lr_2e-5_HH_puyu_mixed_DATA_HH_puyu_mixed_lr_1e-6_batch_1024_epoch_1.sh"

    /cpfs01/shared/public/dlc create job --config /cpfs01/shared/alillm_hs/zouyicheng/dlc.config \
    --name $name \
    --worker_count $num_nodes \
    --kind PyTorchJob \
    --worker_cpu $node_cpus \
    --worker_gpu $num_gpus \
    --worker_memory $node_mems \
    --workspace_id ws1lu4iyv5yjjyvp \
    --data_sources data1ewbw1ztmmyh,data1bgvj0n14to0,data1dfp0cngxv41,data1ubhj4714msc,data1xj7ojru0t4t,data1o8qdjce0kd0 \
    --worker_image pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/pjlab-eflops/lingjun-pytorch-training:2.3-24.03-gu8-gpu-py310-cu124-ubuntu22.04 \
    --worker_shared_memory 256Gi \
    --command "$cmd" \
    --priority 4
}

train_batch_size=1024
lr=1e-6
epoch=1
data_type=Mixed-1252K
# data_type=HH-905K
# policy_model_name=llama3-8B-Instruct
policy_model_name=Qwen2_5_32B
# policy_model_name=InternLM3-8B-Instruct
rm_name=final_RM7B_2e-5_HH_puyu_fixed

commit 
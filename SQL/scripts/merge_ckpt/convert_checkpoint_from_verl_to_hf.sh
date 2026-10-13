python ./model_merger.py merge \
    --backend fsdp \
    --hf_model_path /mnt/workspace/common/models/Qwen2.5-3B \
    --local_dir /mnt/workspace/TreeGRPO/verl_checkpoints/ts_fix-em-max3-format-2_2_1_3-k3kl-inner_inter-step180-normbs/actor/global_step_160 \
    --target_dir /mnt/workspace/TreeGRPO/scripts/merge_ckpt/qwen2.5-ts
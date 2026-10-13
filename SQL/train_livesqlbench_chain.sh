export DATA_DIR='/dataset/wang0702/livesqlbench'
export DB_ENV_URL='http://127.0.0.1:6002'

export SWANLAB_API_KEY="your_swanlab_api_key"
export SWANLAB_MODE="cloud"
export SWANLAB_LOG_DIR="./swanlab_log"
export TENSORBOARD_DIR="./tensorboard_log"
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

export WG_BACKEND="ray"
export RAY_gsc_rpc_server_reconnect_timeout_s=100
export BASE_DIR=$(pwd)

WAND_PROJECT="AutoDenseTreeNew"
N_NODES=1

n_gpus_per_node=4
train_batch_size=$[$n_gpus_per_node * 15]
val_batch_size=$[$n_gpus_per_node * 35]

tree_search_m=10
tree_search_k=1   # ts_n
tree_search_l=0
# leaves_per_prompt = m + l*n = 56 + 0*1 = 56 (no tree expansion -- m independent chains only)
leaves_per_prompt=$[$tree_search_m + $tree_search_l * $tree_search_k]
# actor.ppo_mini_batch_size/ppo_micro_batch_size are both configured in TRAJECTORY units (i.e.
# post tree-expansion rollout count, not prompt count -- see fsdp_workers.py _normalize_config),
# so they're derived from train_batch_size * leaves_per_prompt, not train_batch_size alone.
# 8 mini-batches (i.e. 8 PPO gradient updates) per rollout batch, matching the original
# Tree-GRPO scripts' train_batch_size/8 convention (now expressed in trajectory units).
actor_ppo_mini_batch_size=$[$train_batch_size * $leaves_per_prompt]
actor_ppo_micro_batch_size=$[$n_gpus_per_node / 4]
log_prob_micro_batch_size=$actor_ppo_micro_batch_size

export BASE_MODEL='/dataset/wang0702/models/Qwen3.5-4B'
export EXPERIMENT_NAME=livesqlbench-gspo-qwen3.5-4b-6_0

ulimit -n 65535

python3 -m verl.trainer.main_ppo_format_ts \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=45000 \
    data.max_response_length=8000 \
    data.max_start_length=3000 \
    data.max_obs_length=1600 \
    data.shuffle_train_dataloader=False \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.entropy_checkpointing=true \
    actor_rollout_ref.actor.policy_loss=gspo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$actor_ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size=$actor_ppo_micro_batch_size \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tree_search=true \
    actor_rollout_ref.rollout.ts_m=$tree_search_m \
    actor_rollout_ref.rollout.ts_n=$tree_search_k \
    actor_rollout_ref.rollout.ts_l=$tree_search_l \
    actor_rollout_ref.rollout.reward_mode=base \
    actor_rollout_ref.rollout.expand_mode=random \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    +actor_rollout_ref.rollout.disable_log_stats=true \
    +actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    algorithm.use_kl_in_reward=false \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger="['swanlab', 'console']" \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.val_repeat=3 \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=2 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${BASE_DIR}/verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=20 \
    db_env.url=$DB_ENV_URL \
    2>&1 | tee verl_log/$EXPERIMENT_NAME.log

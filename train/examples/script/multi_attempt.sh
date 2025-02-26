PRETRAIN=${1:-Qwen/Qwen2.5-Math-1.5B}
HDFS_HOME=${2:-large_data}
RUN_NAME=${3:-multi_attempt}

export TOKENIZERS_PARALLELISM=true

# set your wandb key in use_wandb
python3 openrlhf/cli/train_ppo_ray.py \
    --multi_attempt \
    --disable_seq_mean \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 4 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --colocate_actor_ref \
    --colocate_critic_vllm \
    --colocate_vllm_mem 0.5 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --pretrain $PRETRAIN \
    --save_path $HDFS_HOME/checkpoints/$RUN_NAME \
    --micro_train_batch_size 8 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size 1024 \
    --temperature 1 \
    --n_samples_per_prompt 1 \
    --max_samples 100000 \
    --max_epochs 1 \
    --num_episodes 160 \
    --prompt_max_len 1024 \
    --generate_max_len 3000 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 1e-5 \
    --init_kl_coef 0.01 \
    --direct_kl \
    --use_kl_estimator_k3 \
    --lambd 0.99 \
    --prompt_data data/math_level3to5_data.json \
    --input_key input \
    --normalize_reward \
    --flash_attn \
    --adam_offload \
    --gradient_checkpointing \
    --eval_steps 8 \
    --save_steps 8 \
    --save_steps_hf 80 \
    --load_checkpoint \
    --use_wandb '' \
    --wandb_project multi_attempt \
    --wandb_run_name $RUN_NAME \
    --ckpt_path $HDFS_HOME/checkpoints/$RUN_NAME  \
    --eval_path $HDFS_HOME/eval/$RUN_NAME  \
    --max_ckpt_num 2




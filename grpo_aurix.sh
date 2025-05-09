set -x

# export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=XFORMERS

aurix_train_path=data/train.parquet
aurix_test_path=data/test.parquet

# Prepare base model
# model_path=Qwen/Qwen2-7B-Instruct
model_path=Qwen/Qwen2.5-14B-Instruct
python3 extend_model_context.py --model_path=$model_path

# Prepare training data
qa_model="gemini"
negative_chunks=90
python3 -m examples.data_preprocess.aurix --negative_chunks=$negative_chunks --qa_model=$qa_model

# Remove previous logs
rm -rf logs/
mkdir -p logs/

# Prepare training files
train_files="['$aurix_train_path']"
test_files="['$aurix_test_path']"

# Prepare training parameters
batch_size=8
ppo_batch_size=64
input_length=120000
output_length=4000
max_length=$((input_length + output_length))
ppo_max_token_len_per_gpu=$((max_length / 4)) # Decrease if OOM
log_prob_max_token_len_per_gpu=$((max_length / 4)) # Decrease if OOM
ulysses_sequence_parallel_size=4 # Increase if OOM

# Run training
PYTHONPATH=/opt/tiger/open_verl python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.name=all_reward_functions \
    custom_reward_function.path=rewards/reward_reasoning.py \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$batch_size \
    data.max_prompt_length=$input_length \
    data.max_response_length=$output_length \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.rollout.temperature=0.5 \
    actor_rollout_ref.rollout.prompt_length=$input_length \
    actor_rollout_ref.rollout.response_length=$output_length \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_length \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_aurix' \
    trainer.experiment_name='qwen2_7b_qa_reasoning' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 $@
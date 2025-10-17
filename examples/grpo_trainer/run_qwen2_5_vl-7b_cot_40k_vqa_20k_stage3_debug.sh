set -x
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=0

ENGINE=${1:-vllm}
MODEL_NAME=${2:-/mnt/publicdataset/lmms-lab/LLaVA-OneVision-1.5-4B-Instruct}
# MODEL_NAME=${2:-/vlm/peirouliang/checkpoints/qwen25_vl_7b_rl_cot_40k_vqa_20k_stage2_new_format}
DATASET_NAME="mix_llava_cot_40k_llava_next_20k_new_format"
# STAGE优先顺序：第3个命令行参数 > 现有环境变量STAGE > 默认3
STAGE=${3:-${STAGE:-3}}

max_prompt_length=$((1024 * 18))
max_response_length=$((1024 * 18))
# export VERL_LOGITS_LOG_FILE=/vlm/peirouliang/verl/logits_multi10.csv

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/vlm/peirouliang/verl/data/$DATASET_NAME/train.parquet \
    data.val_files=/vlm/peirouliang/verl/data/$DATASET_NAME/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    data.answer_suffix_mode=stage${STAGE} \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=llava_ov_7b_mix_cot_40k_vqa_20k_stage${STAGE} \
    trainer.experiment_name=llava_ov_7b_mix_cot_40k_vqa_20k_stage${STAGE} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    custom_reward_function.path=verl/utils/reward_score/mix_think.py \
    custom_reward_function.name=compute_score

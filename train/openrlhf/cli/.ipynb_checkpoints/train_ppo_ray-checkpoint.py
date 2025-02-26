import argparse
from datetime import datetime
from typing import List
import os

import ray
import torch
from ray.util.placement_group import placement_group
import subprocess
import sys

from openrlhf.trainer.ray import (
    ActorModelRayActor,
    CriticModelRayActor,
    PPORayActorGroup,
    ReferenceModelRayActor,
    DuaReferenceModelRayActor,
    create_vllm_engines,
)
from openrlhf.utils import get_strategy, get_git_version
from openrlhf.duality import dua_create_vllm_engines, MulLLM
from openrlhf import __version__

# NOTE: reward function for multiple reward models, replace this with your own function!
def reward_fn(rewards: List[torch.Tensor]):
    return torch.stack(rewards).sum(dim=0)


def _validate_args(args):
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node

    assert (
        actor_world_size & (actor_world_size - 1)
    ) == 0, f"actor_world_size must be power of 2, got {actor_world_size}"

    if args.critic_pretrain:
        critic_world_size = args.critic_num_nodes * args.critic_num_gpus_per_node
        assert (
            critic_world_size & (critic_world_size - 1)
        ) == 0, f"critic_world_size must be power of 2, got {critic_world_size}"
        assert (
            actor_world_size % critic_world_size == 0
        ), f"actor_world_size must be divisible by critic_world_size, got {actor_world_size} and {critic_world_size}"

    assert args.zero_stage != 3 or args.vllm_num_engines > 0, f"ZeRO-3 is only supported when vLLM enabled"


def train(args):
    _validate_args(args)

    # configure strategy
    strategy = get_strategy(args)
    critic_strategy = get_strategy(args, is_critic=True)

    # if colocated, create placement group for actor and ref model explicitly.    
    has_ref_model = args.duality or args.init_kl_coef > 0.

    if args.colocate_actor_vllm or args.colocate_critic_vllm:
        vllm_num_engines = 0
        if args.colocate_critic_vllm:
            assert args.critic_num_nodes == args.vllm_tensor_parallel_size
            vllm_num_engines += args.critic_num_gpus_per_node
        if args.colocate_actor_vllm:
            assert args.actor_num_nodes == args.vllm_tensor_parallel_size
            vllm_num_engines += args.actor_num_gpus_per_node
        assert vllm_num_engines == args.vllm_num_engines, \
            f"Should have {vllm_num_engines} vllm engine instead of the provided {args.vllm_num_engines}"

    actor_pg = None
    if (args.colocate_actor_ref and has_ref_model) or args.colocate_actor_vllm:
        assert (
            args.actor_num_nodes == args.ref_num_nodes and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

        bundles = [
            {"GPU": args.actor_num_gpus_per_node, "CPU": args.actor_num_gpus_per_node * 4}
            for _ in range(args.actor_num_nodes)
        ]
        actor_pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(actor_pg.ready())

    # 0.75/0.24/0.01 gpu is a tricky to let Ray spread all models evenly on all gpus.
    #   |actor| ref  | vLLM | actor| ref  | vLLM | ...
    #   |GPU0 | GPU0 | GPU0 | GPU1 | GPU1 | GPU1 | ...
    
    num_gpus_per_actor = 1
    if args.colocate_actor_ref and has_ref_model:
        num_gpus_per_actor = num_gpus_per_actor - 0.25
    if args.colocate_actor_vllm:
        num_gpus_per_actor = num_gpus_per_actor - 0.01

    actor_model = PPORayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        ActorModelRayActor,
        pg=actor_pg,
        num_gpus_per_actor=num_gpus_per_actor,
    )

    if has_ref_model:
        ref_model = PPORayActorGroup(
            args.ref_num_nodes,
            args.ref_num_gpus_per_node,
            ReferenceModelRayActor if not args.duality else DuaReferenceModelRayActor,
            pg=actor_pg if args.colocate_actor_ref else None,
            num_gpus_per_actor=0.25 if args.colocate_actor_ref else 1,
        )
    else:
        ref_model = None
        
    # if colocated, create placement group for critic and vllm explicitly.
    critic_pg = None
    if args.critic_pretrain and args.colocate_critic_vllm:
        bundles = [
            {"GPU": args.critic_num_gpus_per_node, "CPU": args.critic_num_gpus_per_node * 2}
            for _ in range(args.critic_num_nodes)
        ]
        critic_pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(critic_pg.ready())

    if args.critic_pretrain:
        critic_model = PPORayActorGroup(
            args.critic_num_nodes,
            args.critic_num_gpus_per_node,
            CriticModelRayActor,
            pg=critic_pg,
            num_gpus_per_actor=0.99 if args.colocate_critic_vllm else 1,
        )
    else:
        critic_model = None
    reward_models = None

    # init reference/reward/actor model
    ray.get(actor_model.async_init_model_from_pretrained(strategy, args.actor_pretrain))
    max_steps = ray.get(actor_model._actor_handlers[0].max_steps.remote())
    refs = []
    refs.extend(critic_model.async_init_model_from_pretrained(critic_strategy, args.critic_pretrain, max_steps))        
    if has_ref_model:
        refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.ref_pretrain, max_steps))    
    if args.skip_sft:
        refs.extend(actor_model.async_save_model())
    ray.get(refs)

    # init vLLM engine for text generation    
    vllm_engines = None
    if args.vllm_num_engines is not None and args.vllm_num_engines > 0:        
        print("Preparing to init vllm engines...")
        max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        pgs = []
        if args.colocate_actor_vllm: pgs.append(actor_pg)
        if args.colocate_critic_vllm: pgs.append(critic_pg)
        vllm_kwargs = dict(
            num_engines=args.vllm_num_engines,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            pretrain=args.actor_pretrain,
            seed=args.seed,
            enable_prefix_caching=args.enable_prefix_caching,
            enforce_eager=args.enforce_eager,
            max_model_len=max_len,
            dtype=args.vllm_dtype,
            pgs=pgs,
        )
        if args.colocate_actor_vllm or args.colocate_critic_vllm:        
            vllm_kwargs["gpu_memory_utilization"] = args.colocate_vllm_mem
            vllm_kwargs["enable_sleep_mode"] = True
        if args.multi_attempt:
            vllm_kwargs["vllm_cls"] = MulLLM
            vllm_kwargs["enable_prefix_caching"] = True
            vllm_kwargs["min_attempt"] = args.min_attempt
            vllm_kwargs["max_attempt"] = args.max_attempt            
            vllm_kwargs["repeat_question"] = args.repeat_question
            vllm_kwargs["attempt_discount"] = args.attempt_discount
            vllm_kwargs["use_hf_math"] = args.use_hf_math        
            vllm_kwargs["custom_sys_prompt"] = args.custom_sys_prompt
            vllm_kwargs["custom_sys_prompt_wrong"] = args.custom_sys_prompt_wrong
            vllm_kwargs["token_per_step"] = args.mul_num_token     
                
        if args.duality:
            vllm_kwargs.pop("pretrain")
            vllm_kwargs["sys1_model"] = args.ref_pretrain
            vllm_kwargs["sys2_model"] = args.actor_pretrain if not args.skip_sft else args.save_path
            vllm_kwargs["sys1_temp"] = args.sys1_temp
            vllm_kwargs["num_sys1_token"] = args.num_sys1_token            
            vllm_kwargs["enable_prefix_caching"] = True
            vllm_engines = dua_create_vllm_engines(**vllm_kwargs)
        else:
            vllm_engines = create_vllm_engines(**vllm_kwargs)

    print("Start training!")
    # train actor and critic mdoel
    refs = actor_model.async_fit_actor_model(
        critic_model, ref_model, reward_models, args.remote_rm_url, reward_fn=reward_fn, vllm_engines=vllm_engines
    )
    ray.get(refs)

    # save model
    ray.get(actor_model.async_save_model())

    if args.critic_pretrain and args.save_value_network:
        ray.get(critic_model.async_save_model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    

    # Duality-agent-specific args
    parser.add_argument("--duality", action="store_true", default=False, help="SFT on dual-system agent")
    parser.add_argument("--ref_learning_rate", type=float, default=1e-6)
    parser.add_argument("--ref_start_step", type=int, default=0, help="Step where ref model starts training")
    parser.add_argument("--sys1_critic", action="store_true", default=False)
    parser.add_argument("--sys1_temp", type=float, default=0.6, help="System 1 sampling temperature")
    parser.add_argument("--num_sys1_token", type=int, default=64, help="Length of System 1 intuition token")
    parser.add_argument("--skip_sft", action="store_true", default=False)    

    # Mult-attempt args
    parser.add_argument("--multi_attempt", action="store_true", default=False, help="Multi-trial environment")
    parser.add_argument("--min_attempt", type=int, default=1, help="Minimum number of possible attempt")
    parser.add_argument("--max_attempt", type=int, default=5, help="Maximum number of possible attempt")
    parser.add_argument("--repeat_question", action="store_true", default=False, help="Repeat the question if wrong")
    parser.add_argument("--attempt_discount", type=float, default=1., help="Discount to reward for each attempt")
    parser.add_argument("--custom_sys_prompt_wrong", type=str, default=None, help="Custom sys prompt template when answered wrong")
    parser.add_argument("--mul_num_token", type=int, default=64, help="Number of token per checking")

    # Ray and vLLM
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference")
    parser.add_argument(
        "--colocate_actor_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and actor model, if true, they will share same gpus.",
    )

    parser.add_argument("--actor_num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8, help="number of gpus per node for actor")
    parser.add_argument("--critic_num_nodes", type=int, default=1, help="number of nodes for critic")
    parser.add_argument("--critic_num_gpus_per_node", type=int, default=8, help="number of gpus per node for critic")
    parser.add_argument(
        "--colocate_critic_reward",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, they will share same gpus.",
    )

    # optional vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl", help="DeepSpeed -> vLLM weight sync backend")
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False, help="Disable CUDA graph in vLLM")
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16", help="data type for vLLM")
    
    parser.add_argument("--colocate_actor_vllm", action="store_true", default=False, help="Colocate actor and vllm together sharing the same GPU.")
    parser.add_argument("--colocate_critic_vllm", action="store_true", default=False, help="Colocate critic and vllm together sharing the same GPU.")
    parser.add_argument("--colocate_vllm_mem", type=float, default=0.5, help="Percentage of GPU used by vLLM")

    # Checkpoints
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=-1, help="Save frequency of Zero checkpoint")
    parser.add_argument("--save_steps_hf", type=int, default=5, help="Save frequency of HF checkpoint")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo_ray")
    parser.add_argument("--max_ckpt_num", type=int, default=2, help="Max number of Zero checkpoint")
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8, help="Max file size of Zero checkpoint")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    ## Make EMA as an optional feature
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # PPO
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--eval_path", type=str, default="./eval")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=1024)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--critic_train_batch_size", type=int, default=-1, help="Global training batch size for critic; negative for being the same as actor")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--constant_warmup", action="store_true", default=False, help="Use constant warm up in scheduler")
    parser.add_argument("--num_warmup_steps", type=int, default=50, help="Number of warm up step in constant warm up")
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument("--direct_kl", action="store_true", default=False, help="Directly optimize KL-div instead of giving it as rewards")
    parser.add_argument(
        "--use_kl_estimator_k3",
        action="store_true",
        default=False,
        help=(
            "Use the k3 estimator in http://joschu.net/blog/kl-approx.html"
            "to ensure the KL divergence calculated is non-negative"
        ),
    )
    parser.add_argument("--actor_value_coef", type=float, default=0, help="Coef for predicting value loss on actor")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")
    parser.add_argument("--disable_normalize_adv", action="store_true", default=False, help="Disabling normalizing advantage")
    parser.add_argument("--disable_seq_mean", action="store_true", default=False, help="Disabling computing sequence mean before averaging loss")

    # Reinforce
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce",
    )

    #  Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path for actor and ref")
    parser.add_argument("--actor_pretrain", type=str, default=None, help="HF model name or path for actor; default to pretrain")
    parser.add_argument("--ref_pretrain", type=str, default=None, help="HF model name or path for ref; default to pretrain")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API (HTTP)")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="value_head")
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")

    parser.add_argument("--input_key", type=str, default=None, help="JSON dataset key for input")
    parser.add_argument("--output_key", type=str, default=None, help="JSON dataset key for answer")
    parser.add_argument("--add_sys_prompt", action="store_true", default=False, help="Manually wrapping input with system prompts")
    parser.add_argument("--custom_sys_prompt", type=str, default=None, help="Custom sys prompt template")
    parser.add_argument("--use_hf_math", action="store_true", default=False, help="Use math-verify instead of qwen to check math equal")
    

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)

    args = parser.parse_args()

    if args.actor_pretrain is None:
        args.actor_pretrain = args.pretrain    
    if args.ref_pretrain is None:
        args.ref_pretrain = args.pretrain   
    if args.critic_pretrain is None:
        args.critic_pretrain = args.actor_pretrain         
    if args.advantage_estimator not in ["gae"]:
        args.critic_pretrain = None
    if args.remote_rm_url:
        args.remote_rm_url = args.remote_rm_url.split(",")

    if args.vllm_num_engines >= 1 and args.enable_prefix_caching:
        args.enable_prefix_caching = False
        print("[Warning] Disable prefix cache because vLLM updates weights without updating the old KV Cache.")

    if args.critic_train_batch_size <= 0:
        args.critic_train_batch_size = args.train_batch_size

    if args.custom_sys_prompt:
        assert '{input}' in args.custom_sys_prompt, f"{{input}} not in {args.custom_sys_prompt}"

    if args.packing_samples:
        if not args.flash_attn:
            print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
            args.flash_attn = True
        assert args.vllm_num_engines > 0, "Only support `--packing_samples` with vLLM."
        assert not args.pretrain_data, "`--pretrain_data` is not supported with `--packing_samples` yet."

    if args.multi_attempt:
        assert not args.add_sys_prompt, "system prompt will be manually added in vllm"
    
    args.cmd = " ".join(sys.argv)
    args.git_version = get_git_version()
    args.version = __version__

    train(args)

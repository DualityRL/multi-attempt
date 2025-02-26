import math
import os
import os.path
import socket
import json
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import timedelta
import time

import deepspeed
import ray
import torch
import torch.nn as nn
from torch import distributed as dist
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
#from tqdm import tqdm

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss
from openrlhf.models.utils import masked_mean, compute_approx_kl
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.trainer.ppo_utils import Experience, RemoteExperienceMakerBOX
from openrlhf.utils.distributed_util import init_process_group
from openrlhf.utils import tqdm
import openrlhf.utils.utils as utils

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveReplayBuffer


class PPOTrainer(ABC):
    """
    Trainer for Proximal Policy Optimization (PPO) algorithm.

    Args:
        strategy (Strategy): The training strategy to use.
        actor (Actor): The actor model in the PPO algorithm.
        critic (nn.Module): The critic model in the PPO algorithm.
        reward_model (nn.Module): The reward model for calculating rewards in the RLHF setup.
        initial_model (Actor): The initial model for reference logits to limit actor updates in RLHF.
        ema_model (Actor): The exponential moving average model for stable training.
        actor_optim (Optimizer): The optimizer for the actor model.
        critic_optim (Optimizer): The optimizer for the critic model.
        actor_scheduler (Scheduler): The learning rate scheduler for the actor.
        critic_scheduler (Scheduler): The learning rate scheduler for the critic.
        ema_beta (float, defaults to 0.992): EMA decay rate for model stability.
        init_kl_coef (float, defaults to 0.001): Initial coefficient for KL divergence.
        kl_target (float, optional): Target value for KL divergence.
        kl_horizon (int, defaults to 10000): Horizon for KL annealing.
        ptx_coef (float, defaults to 0): Coefficient for supervised loss from pre-trained data.
        micro_train_batch_size (int, defaults to 8): Micro-batch size for actor training.
        buffer_limit (int, defaults to 0): Maximum size of the replay buffer.
        buffer_cpu_offload (bool, defaults to True): If True, offloads replay buffer to CPU.
        eps_clip (float, defaults to 0.2): Clipping coefficient for policy loss.
        value_clip (float, defaults to 0.2): Clipping coefficient for value function loss.
        micro_rollout_batch_size (int, defaults to 8): Micro-batch size for generating rollouts.
        gradient_checkpointing (bool, defaults to False): If True, enables gradient checkpointing.
        max_epochs (int, defaults to 1): Number of epochs to train.
        max_norm (float, defaults to 1.0): Maximum gradient norm for gradient clipping.
        tokenizer (Callable, optional): Tokenizer for input data.
        prompt_max_len (int, defaults to 128): Maximum length for prompts.
        dataloader_pin_memory (bool, defaults to True): If True, pins memory in the data loader.
        remote_rm_url (str, optional): URL for remote reward model API.
        reward_fn (Callable, optional): Custom reward function for computing rewards.
        **generate_kwargs: Additional arguments for model generation.
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List = None,
        critic_train_remote: bool = False,
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)
        self.start_train_ref = False

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        packing_samples = getattr(self.args, "packing_samples", False)
        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size, buffer_limit, buffer_cpu_offload, packing_samples
        )

        self.remote_rm_url = remote_rm_url
        self.vllm_engines = vllm_engines
        self.critic_train_remote = critic_train_remote

        self.experience_maker = RemoteExperienceMakerBOX(
            self.actor,
            self.critic,
            self.reward_model,
            self.initial_model,
            self.tokenizer,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.remote_rm_url,
            self.reward_fn,
            vllm_engines=self.vllm_engines,
            packing_samples=self.strategy.args.packing_samples,
        )


        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
            # https://github.com/OpenRLHF/OpenRLHF/issues/313
            import vllm

            if vllm.__version__ > "0.4.2" and os.getenv("NCCL_P2P_DISABLE", "0") == "0":
                backend = "gloo"
                ray.logger.info(
                    "Warning: using --vllm_sync_backend=gloo for vLLM version > 0.4.2 (or export NCCL_P2P_DISABLE=1)"
                )
            
            group_name = "openrlhf"
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]            
            ray.logger.info(f"Start init_process_group rank:{0}/{world_size} group_name: {group_name} tcp://{master_address}:{master_port}")
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=group_name,
                timeout=timedelta(seconds=7200),
            )
            ray.logger.info(f"Finish init_process_group rank:{0}/{world_size} group_name: {group_name} tcp://{master_address}:{master_port}")
            ray.get(refs)            

        torch.distributed.barrier()

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb
            self._wandb = wandb
            if not wandb.api.api_key:                
                wandb.login(key=strategy.args.use_wandb)            
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                id=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                resume="allow",
                allow_val_change=True,
                settings=wandb.Settings(init_timeout=3600),
            )
            #wandb.config.update(strategy.args.__dict__, allow_val_change=True)

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/*", step_metric="train/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(
        self,
        args,
        prompts_dataloader,
        pretrain_dataloader,
        consumed_samples=0,
        best_eval_score=float('-inf'),
        num_update_steps_per_episodes=1,
    ) -> None:
        num_rollouts_per_episodes = (
            num_update_steps_per_episodes
            * args.train_batch_size
            // args.max_epochs
            // args.rollout_batch_size
            // args.n_samples_per_prompt
        )

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt
        self.best_eval_score = best_eval_score

        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts in self.prompts_dataloader:
                rand_targets = rand_prompts["target"]
                rand_answer = rand_prompts["answer"]
                rand_prompts = rand_prompts["input"]                

                for i, experience in enumerate(
                    self.experience_maker.make_experience_list(rand_prompts, rand_answer, **self.generate_kwargs)
                ):
                    if i == 0:
                        output = self.tokenizer.batch_decode(
                            experience.sequences[0].unsqueeze(0), skip_special_tokens=False
                        )
                        #self.strategy.print(output[0].replace("<|endoftext|>", ""))
                        self.strategy.print(output[0].replace("<|endoftext|>", "").encode('utf-8', 'ignore').decode('utf-8'))

                    self.replay_buffer.append(experience)

                if args.colocate_actor_ref and self.initial_model is not None:
                    ray.get(self.initial_model.sleep.remote())

                if self.strategy.is_rank_0():            
                    ray.logger.info(f"\033[92m[{self.args.wandb_run_name}] Starting 3. PPO training...\033[0m") 
                    start_time = time.time()

                torch.distributed.barrier()
                torch.cuda.empty_cache()
                if not args.disable_normalize_adv:
                    self.replay_buffer.normalize("advantages", self.strategy)
                status = self.ppo_train(steps)
                self.replay_buffer.clear()
                torch.cuda.empty_cache()

                if self.strategy.is_rank_0():            
                    end_time = time.time()  # Record end time
                    elapsed_time = (end_time - start_time) / 60  # Convert to minutes
                    ray.logger.info(f"\033[92m[{self.args.wandb_run_name}] Finished 3. PPO training in {elapsed_time:.2f} minutes\033[0m")


                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
                pbar.set_postfix(status)

                # logs/checkpoints
                client_states = {"consumed_samples": steps * args.rollout_batch_size, "best_eval_score": self.best_eval_score}
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                pbar.update()
                steps = steps + 1

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()
        with open(os.path.join(args.ckpt_path, ".complete"), "w") as file:
            pass  # Creates an empty file

    def ppo_train(self, global_steps):
        # 1. ensure all experience makers done
        self.experience_maker.flush()
        torch.distributed.barrier()

        # 2. trigger remote critic model training
        refs = []
        if self.critic_train_remote:
            refs.append(self.critic.fit.remote())

        # 3. actor model training
        if global_steps > self.freezing_actor_steps:
            status = self.ppo_loop(global_steps)
        else:
            status = {}

        # 4. wait remote critic /ref model training done
        if len(refs) > 0:
            sub_status = ray.get(refs)
            for s in sub_status: status.update(s)

        # 5. broadcast weights to vllm engines
        if self.vllm_engines is not None:
            torch.distributed.barrier()
            self._broadcast_to_vllm(sleep=False)

        return status            

    def ppo_loop(self, global_steps=0):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience)

                status["kl"] *= status["response_length"]
                status = self.strategy.all_reduce(status)
                status["kl"] /= status["response_length"]

                if self.args.multi_attempt:
                    prefixes = ["attempt_used_success", "attempt_used_failure"]
                    for prefix in prefixes:
                        _sum = status.pop(prefix+"_s")
                        _count = status.pop(prefix+"_c")
                        if _count > 0:
                            status[prefix] = _sum / _count
                        else:
                            status[prefix] = 0.

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "ret": status["return"],
                        "glen": status["response_length"],
                        "tlen": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                    }

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)
                torch.cuda.empty_cache()

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean
    
    def training_step(self, experience: Experience) -> Dict[str, float]:
        # actor training; critic is in another ray-actor
        self.actor.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            base_action_log_probs = torch.cat(experience.base_action_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            returns = torch.cat(experience.returns, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_action_log_probs = experience.action_log_probs
            base_action_log_probs = experience.base_action_log_probs
            advantages = experience.advantages
            returns = experience.returns
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask

        # actor loss
        action_log_probs, output = self.actor(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )        

        # loss function
        mask = experience.action_mask
        if self.args.multi_attempt:
            mask = torch.logical_and(mask, torch.logical_not(experience.sys2_mask))
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=mask,
            disable_seq_mean=self.args.disable_seq_mean,
        )
        
        # eps in PPO
        eps = torch.abs((action_log_probs - old_action_log_probs).exp() - 1)
        mean_eps = masked_mean(eps, mask, dim=-1).mean()
        clamp_ratio = (eps >= self.actor_loss_fn.clip_eps).float()
        mean_clamp_ratio = masked_mean(clamp_ratio, mask, dim=-1).mean()

        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = actor_loss + aux_loss * self.args.aux_loss_coef
        # direct kl
        if self.args.direct_kl and self.args.init_kl_coef > 0.:
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=mask,
                use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
            )
            if self.args.disable_seq_mean:
                kl = masked_mean(kl, mask)
            else:
                kl = masked_mean(kl, mask, dim=-1).mean()
            loss = loss + self.args.init_kl_coef * kl

        if self.args.actor_value_coef > 0.:
            returns = returns.to(dtype=output["values"].dtype)
            #actor_value_loss = (output["values"] - returns)**2
            actor_value_loss = nn.functional.huber_loss(output["values"], returns, reduction='none', delta=1.0)
            actor_value_loss = 0.5 * masked_mean(actor_value_loss, mask, dim=-1).mean()
            loss = loss + self.args.actor_value_coef * actor_value_loss

        self.strategy.backward(loss, self.actor, self.actor_optim)

        # ptx loss
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
            ptx_log_probs = output["logits"]

            # loss function
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # mixtral
            if self.aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")        

        # status
        status = {"policy_loss": actor_loss.item(), 
                  "actor_lr": self.actor_scheduler.get_last_lr()[0], 
                  "ppo_eps": mean_eps.item(), 
                  "ppo_clamp": mean_clamp_ratio.item(),
                  }

        if self.args.actor_value_coef > 0.:
            status["actor_value_loss"] = actor_value_loss.item()

        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()
        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.float().mean().item()

        if self.args.multi_attempt:
            attempt_used = experience.info["attempt_used"].float()
            reward = experience.info["reward"]
            status["attempt_used_success_s"] = torch.sum(attempt_used * (reward > 0.)).item()
            status["attempt_used_success_c"] = torch.sum(reward > 0.).float().item()
            status["attempt_used_failure_s"] = torch.sum(attempt_used * (reward <= 0.)).item()
            status["attempt_used_failure_c"] = torch.sum(reward <= 0.).float().item()
        return status

    def _broadcast_to_vllm(self, sleep=False):
        ray.logger.info(f"\033[92m Broadcasting weight to vLLM...\033[0m")                    
        # avoid OOM
        torch.cuda.empty_cache()
        model = self.actor.model.module            
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param
            if name.startswith("ac_value_head"): continue

            # Fire all vllm engines for broadcast
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(
                        name, 
                        dtype=param.dtype, 
                        shape=shape, 
                        empty_cache=count == num_params,
                        sleep=(count == num_params) and sleep,
                    )
                    for engine in self.vllm_engines
                ]

            # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
            with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                if torch.distributed.get_rank() == 0:
                    torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                    ray.get(refs)
        torch.distributed.barrier()    
        
    def evaluate(self, args, tag):
        output_dir = os.path.join(args.eval_path, tag)

        if self.strategy.is_rank_0():            
            ray.logger.info(f"\033[92m[{self.args.wandb_run_name}] Starting 4. Evaluation {tag}...\033[0m") 
            start_time = time.time()                

        results = utils.vllm_math_evaluate(self.vllm_engines, output_dir, args=args, enable_sleep=False)

        if self.strategy.is_rank_0():            
            end_time = time.time()  # Record end time
            elapsed_time = (end_time - start_time) / 60  # Convert to minutes
            ray.logger.info(f"\033[92m[{self.args.wandb_run_name}] Finished 4. Evaluation {tag} in {elapsed_time:.2f} minutes with acc {results['eval/avg_acc']:.2f} \033[0m")

        return results
            
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        eval_results = None 
        tag = f"global_step{global_step}"

        if self.strategy.is_rank_0():
            if global_step % args.eval_steps == 0:
                eval_results = self.evaluate(args, tag)
            if global_step % args.logging_steps == 0:
                # wandb
                if self._wandb is not None:
                    logs = {
                        "train/%s" % k: v
                        for k, v in {
                            **logs_dict,
                            "global_step": global_step,
                        }.items()
                    }
                    if eval_results is not None:
                        logs.update(eval_results)
                    if self.experience_maker.perf_stats is not None:
                        logs.update({f"perf/experience_maker/{k}": v for k, v in self.experience_maker.perf_stats.items()})
                    self._wandb.log(logs)
                # TensorBoard
                elif self._tensorboard is not None:
                    for k, v in logs_dict.items():
                        self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                    if eval_results is not None:
                        for k, v in eval_results.items():
                            self._tensorboard.add_scalar(f"eval/{k}", v, global_step)
                    if self.experience_maker.perf_stats is not None:
                        for k, v in self.experience_maker.perf_stats.items():
                            self._tensorboard.add_scalar(f"perf/experience_maker/{k}", v, global_step)
        
        # broadcast eval_results from rank 0 to all ranks
        sync_eval_results = [None] * self.strategy.world_size
        dist.all_gather_object(sync_eval_results, eval_results)
        eval_results = sync_eval_results[0]
        
        if global_step % args.save_steps == 0:            
            self._save_checkpoint(args, tag, client_states)

        if global_step % args.save_steps_hf == 0:
            self._save_hf(args, tag, global_step)

        # save the best hf model
        if eval_results is not None:
            eval_score = eval_results["eval/avg_acc"]
            if eval_score is not None and eval_score > self.best_eval_score:
                self.best_eval_score = eval_score
                self._save_hf(args, "best", global_step)

    def _save_checkpoint(self, args, tag, client_states):
        # save critic
        refs = []
        if self.critic_train_remote:
            refs.append(self.critic.save_checkpoint.remote(tag))

        # Original checkpoint saving for training resumption
        self.strategy.save_ckpt(
            self.actor.model,
            os.path.join(args.ckpt_path, "_actor"),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )
        if len(refs) > 0: ray.get(refs)  

    def _save_hf(self, args, tag, step):
        # Save in Hugging Face format        
        save_path = os.path.join(args.ckpt_path.rstrip("/"), "_actor_hf", tag)            
        if self.strategy.is_rank_0(): 
            ray.logger.info(f"Saving hugging face models to {save_path}...")    

        self.strategy.save_model(self.actor, self.tokenizer, save_path)
                  
        if self.strategy.is_rank_0(): 
            # Save training arguments
            args_dict = vars(args)
            args_dict["global_step"] = step
            # Save to JSON file
            with open(os.path.join(save_path, "args.json"), "w") as f:
                json.dump(args_dict, f, indent=4)
            ray.logger.info(f"Finish saving hugging face models to {save_path}...")

import logging
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.nn as nn

import torch
import torch.nn.functional as F
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.models.actor import Actor
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils import parallel_f, tqdm, save_debug_data
from openrlhf.datasets.utils import create_token_mask
from openrlhf.trainer.ppo_utils.qwen_math_eval_toolkit.grader import math_equal, math_equal_hf
from openrlhf.trainer.ppo_utils.qwen_math_eval_toolkit.parser import extract_answer
import re

logger = init_logger(__name__)

def preprocess_box_response_for_qwen_prompt(queries, answers, use_hf_math=False):
    
    extract_answers = []
    contain_box_checks = []
    for query in queries:
        model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', query, flags=re.DOTALL,count = 1)
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
        for stop_word in stop_words:
            if stop_word in model_output:
                model_output = model_output.split(stop_word)[0].strip()
        extract_answers.append(extract_answer(model_output, data_name="math"))
        contain_box_checks.append("boxed" in model_output)
    f = math_equal_hf if use_hf_math else math_equal
    corrects = parallel_f(f, list(zip(extract_answers, answers)), num_workers=16, timeout=10, default_value=False)
    
    reward = []    
    for contain_box, correct in zip(contain_box_checks, corrects):
        if correct:
            reward.append(1.)
        elif contain_box:
            reward.append(-0.5)
        else:
            reward.append(-1.)
    
    return torch.tensor(reward)

def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)

def fill_left_with_ones(mask):
    M, B = mask.shape
    first_one_idx = (mask == 1).to(torch.int).argmax(dim=1)  # Shape (M,)
    indices = torch.arange(B, device=mask.device).expand(M, B)
    fill_mask = indices < first_one_idx.unsqueeze(1)  # Shape (M, B)
    mask = mask.clone()  # Avoid modifying in-place
    mask[fill_mask] = 1
    return mask

def convert_action_item_to_sys1(item, sys1_action_mask, sys2_action_mask, sys2_mask):
    if item is None: return None
    x = torch.zeros(sys1_action_mask.shape, dtype=item.dtype, device=item.device)
    valid_mask = torch.logical_and(torch.logical_not(sys2_mask), sys2_action_mask)
    x[sys1_action_mask] = item[valid_mask]
    return x

def compute_sys1_action_mask(sys2_action_mask, sys2_mask):
    device = sys2_action_mask.device
    valid_mask = torch.logical_and(torch.logical_not(sys2_mask), sys2_action_mask)
    valid_len = torch.sum(valid_mask, dim=1)
    max_len = valid_len.max().item()
    range_tensor = torch.arange(max_len, device=device).unsqueeze(0)
    sys1_action_mask = (range_tensor < valid_len.unsqueeze(1)).bool()
    return sys1_action_mask

def convert_sequences_to_sys1(sequences, attention_mask, action_mask, sys2_mask, tokenizer):    
    device = sequences.device
    mask_device = action_mask.device
    action_mask = action_mask.to(device)
    sys2_mask = sys2_mask.to(device)
    attention_mask = attention_mask.to(device)

    input_len = sequences.shape[1] - action_mask.shape[1]
    num_actions = action_mask.shape[1]

    sys1_action_mask = compute_sys1_action_mask(action_mask, sys2_mask)
    valid_mask = torch.logical_and(action_mask, torch.logical_not(sys2_mask))    
    
    sys1_responses = torch.full(sys1_action_mask.shape, dtype=sequences.dtype, device=device, fill_value=tokenizer.pad_token_id)
    sys1_responses[sys1_action_mask] = sequences[:, -num_actions:][valid_mask]
    sys1_sequences = torch.cat([sequences[:, :-num_actions], sys1_responses], dim=1)

    sys1_attention_mask = torch.cat([attention_mask[:, :input_len], sys1_action_mask], dim=1)
    #sys1_attention_mask[sys1_sequences == tokenizer.eos_token_id] = 0

    return sys1_sequences, sys1_attention_mask.to(mask_device), sys1_action_mask.to(mask_device), valid_mask.to(mask_device)

def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device)

def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory()

@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    base_action_log_probs: (B, A)
    action_log_probs: (B, A)        
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor    
    base_action_log_probs: torch.Tensor
    action_log_probs: torch.Tensor    
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    sys2_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        if self.values is not None:
            self.values = to(self.values, device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)
        if self.sys2_mask is not None:
            self.sys2_mask = self.sys2_mask.to(device)            

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        if self.values is not None:
            self.values = pin_memory(self.values)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        if self.sys2_mask is not None:
            self.sys2_mask = self.sys2_mask.pin_memory()            
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    
    
@dataclass
class SamplesBOX:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    sys2_mask:Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    answers: List[str]    
    rewards: Optional[torch.Tensor]
    attempt_used: Optional[torch.Tensor]
    
class RemoteExperienceMakerBOX(ABC):
    def __init__(
            self, 
            actor: Actor,
            critic: nn.Module,
            reward_model: nn.Module,
            initial_model: Actor,
            tokenizer,
            prompt_max_len: int,
            kl_controller,
            strategy=None,
            remote_rm_url: str = None,
            reward_fn=None,            
            vllm_engines: List = None, 
            packing_samples=False
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples
        self.send_to_ref = False

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], all_answers:  Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }

        args = self.strategy.args
        experiences = []
        sys1_experiences = []
        
        if self.strategy.is_rank_0():            
            ray.logger.info(f"\033[92m[{self.strategy.args.wandb_run_name}] Starting 1. Prompt generation...\033[0m") 
            start_time = time.time()

        all_samples = self.generate_samples(all_prompts, all_answers, **generate_kwargs)

        if self.strategy.is_rank_0():            
            end_time = time.time()  # Record end time
            elapsed_time = (end_time - start_time) / 60  # Convert to minutes
            ray.logger.info(f"\033[92m[{self.strategy.args.wandb_run_name}] Finished 1. Prompt generation in {elapsed_time:.2f} minutes\033[0m")

        if self.strategy.is_rank_0():            
            ray.logger.info(f"\033[92m[{self.strategy.args.wandb_run_name}] Starting 2. Make experience...\033[0m") 
            start_time = time.time()

        all_samples = tqdm(
            all_samples,
            desc=f"make_experience",
            disable=not self.strategy.is_rank_0(),
        )
        for idx, samples in enumerate(all_samples):
            experience = self.make_experience(samples, **generate_kwargs)
            experiences.append(experience)

        if self.strategy.is_rank_0():            
            end_time = time.time()  # Record end time
            elapsed_time = (end_time - start_time) / 60  # Convert to minutes
            ray.logger.info(f"\033[92m[{self.strategy.args.wandb_run_name}] Finished 2. Make experience in {elapsed_time:.2f} minutes\033[0m")

        # experiences = self.process_experiences(experiences)

        # calculate return and advantages
        for experience in experiences:
            num_actions = experience.info["num_actions"]
            #ray.logger.info(f"rewad shape {experience.info['reward'].shape} kl shape {experience.kl.shpae} action_mask shape {experience.action_mask.shape}")
            reward = compute_reward(
                experience.info["reward"],
                self.kl_ctl.value if not self.strategy.args.direct_kl else 0.,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    experience.sys2_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                    experience.sequences,
                )
            elif self.advantage_estimator == "reinforce":
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums            
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]    

        if self.critic is not None:
            critic_experiences = experiences
            for experience in critic_experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences
    
    def set_send_to_ref(self, enabled: bool):
        self.send_to_ref = enabled

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_answers: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return self._generate_no_vllm(all_prompts, all_answers, **generate_kwargs)

        out = self._generate_vllm(all_prompts, all_answers, **generate_kwargs)    

        if self.strategy.args.colocate_actor_vllm or self.strategy.args.colocate_critic_vllm:
            torch.distributed.barrier()
            if self.strategy.is_rank_0():
                ray.get([v.sleep.remote() for v in self.vllm_engines])

        torch.cuda.empty_cache()
        if self.initial_model is not None:
            ray.get(self.initial_model.empty_cache.remote())
        torch.distributed.barrier()
        
        return out
    
    @torch.no_grad()
    def make_experience(self, samples: SamplesBOX, **generate_kwargs) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        sys2_mask = samples.sys2_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens
        answers = samples.answers

        base_action_log_probs = None
        sys1_sequences = None

        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )
        #ray.logger.info(f"sequences_len: {sequences_cpu.shape}")

        # value
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )

            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)
        #ray.logger.info("sent critic model forward request")

        # init log probs
        if self.initial_model is not None and self.strategy.args.init_kl_coef > 0.:
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            if self.strategy.args.colocate_actor_ref:
                ray.get([self.initial_model.empty_cache.remote()])                    
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
                torch.distributed.barrier()
        #ray.logger.info("sent initial model forward request")

        # rewards
        assert not self.remote_rm_url, NotImplementedError()            

        if self.strategy.args.multi_attempt:
            reward = samples.rewards.to(device=device)  
        else:
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)      
            reward = preprocess_box_response_for_qwen_prompt(queries, answers, use_hf_math=self.strategy.args.use_hf_math)
            reward = reward.to(device=device)              

        #ray.logger.info("calling actor forward")
        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        actor_value_rm_time = time.time() - start
        #ray.logger.info("finish cal actor log probs")
        # wait initial/critic/reward model done     

        start = time.time()
        refs = [value_ref]
        if self.initial_model is not None and self.strategy.args.init_kl_coef > 0.:
            refs.append(base_action_log_probs_ref)
        ref_values = ray.get(refs)
        wait_time = time.time() - start
        #ray.logger.info("finish waiting all models")

        value = ref_values[0]        
        if value is not None:
            value = value.to(device)
        
        if self.initial_model is not None and self.strategy.args.init_kl_coef > 0.:
            base_action_log_probs = ref_values[1]
            base_action_log_probs = base_action_log_probs.to(device)
            mask = action_mask

            if self.strategy.args.multi_attempt:
                mask = torch.logical_and(mask, torch.logical_not(sys2_mask))

            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=mask,
                use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
            )
            # ray.logger.info("finish cal kl")
            if not self.packing_samples:
                kl_mean = masked_mean(kl, action_mask, dim=-1)
            else:
                # convert tensor into list of tensors so that it's easier to manipulate
                # within dataset.
                sequences = unpacking_samples(sequences, packed_seq_lens)
                attention_mask = None
                action_log_probs = unpacking_samples(action_log_probs, num_actions)
                if value is not None:
                    value = unpacking_samples(value, num_actions)

                kl = unpacking_samples(kl, num_actions)
                kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)     
        else:            
            kl = torch.zeros_like(action_log_probs)
            if not self.packing_samples:
                kl_mean = kl[..., 0]
            else:
                kl = unpacking_samples(kl, num_actions)
                kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)    

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()       

        info = {
            "kl": kl_mean,
            "reward": reward,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        if self.strategy.args.multi_attempt:
            info["attempt_used"] = samples.attempt_used.to(device)

        experience = Experience(
            sequences,
            base_action_log_probs,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            sys2_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
        return experience

    @torch.no_grad()
    def _generate_no_vllm(self, all_prompts: List[str], all_answers: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        assert not self.strategy.args.multi_attempt
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_answers = sum([[answer] * args.n_samples_per_prompt for answer in all_answers], [])
        samples_list = []
        #self.strategy.print(f"generate samples!!!")
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            answers = all_answers[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            #self.strategy.print(f"generating!!!")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            #self.strategy.print(f"generating!!!", sequences)
            samples = SamplesBOX(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                sys2_mask=None,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                answers=answers,
            )
            samples_list.append(samples)
            #print("sequences", samples.sequences)            
            #print("attention_mask", samples.attention_mask)
        return samples_list

    def _generate_vllm(self, all_prompts: List[str], all_answers: List[str], **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args
        # print("For debugging:", "qwen" in self.strategy.args.pretrain.lower())
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            stop=["</s>", "<|im_end|>", "<|endoftext|>"] if "qwen" in self.strategy.args.ref_pretrain.lower() else [],
            stop_token_ids=[151645, 151643] if "qwen" in self.strategy.args.ref_pretrain.lower() else [],
            include_stop_str_in_output=True,
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_answers = sum([[answer] * args.n_samples_per_prompt for answer in all_answers], [])

        if not args.multi_attempt:
            all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]
            # Distribute requests to engines and collect responses to outputs
            all_output_refs = []
            batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
            for i, llm in enumerate(llms):
                prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
                if prompt_token_ids:
                    all_output_refs.append(
                        llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                    )
        else:
            # dua / mul vllm does not support generate based on prompt_token_ids yet
            all_output_refs = []
            batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
            for i, llm in enumerate(llms):
                prompts = all_prompts[i * batch_size : (i + 1) * batch_size]
                answers = all_answers[i * batch_size : (i + 1) * batch_size]
                if prompts:
                    if not args.multi_attempt:
                        refs = llm.generate.remote(prompts, sampling_params=sampling_params)
                    else:
                        refs = llm.generate.remote(prompts, sampling_params=sampling_params, answer=answers)
                    all_output_refs.append(refs)

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            answers = all_answers[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                action_mask = []
                attention_mask = []
                sys2_mask = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)
                    attention_mask.append([0] * (max_input_len - input_len) + [1] * len(output.prompt_token_ids))

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)
                    action_mask.append([1] * len(output.outputs[0].token_ids) + [0] * (max_output_len - output_len))                    

                    #if output_ids[output_len - 1] != eos_token_id:
                    #    output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id

                    # concat input and output
                    ids = input_ids + output_ids                    
                    sequences.append(ids)
                    if args.multi_attempt: 
                        start_token="<|im_end|>\n<|im_start|>system"
                        end_token="<|im_start|>assistant\n"
                        sys2_mask.append(create_token_mask(output_ids, self.tokenizer, start_token=start_token, end_token=end_token))

                sequences = torch.tensor(sequences)                
                action_mask = torch.tensor(action_mask)
                
                attention_mask = torch.tensor(attention_mask)
                attention_mask = torch.cat([attention_mask, action_mask], dim=1)
                #attention_mask[sequences == self.tokenizer.eos_token_id] = 0
                #sequences, attention_mask, action_mask = process_sequences(
                #    sequences, max_input_len, eos_token_id, pad_token_id
                #)               

                if args.multi_attempt:
                    sys2_mask = torch.tensor(sys2_mask, dtype=torch.bool).to("cuda")                    
                    #ray.logger.info(f"sys2_mask: {sys2_mask.cpu().sum().item()} out of {sys2_mask.numel()}")        
                else:
                    sys2_mask = None                         

                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")

                if args.multi_attempt:
                    rewards = [output.reward for output in outputs]
                    rewards = torch.tensor(rewards, dtype=torch.float).to("cuda")
                    attempt_used = [output.attempt_used for output in outputs]
                    attempt_used = torch.tensor(attempt_used, dtype=torch.long).to("cuda")
                else:
                    rewards = None
                    attempt_used = None

                samples_list.append(
                    SamplesBOX(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        sys2_mask=sys2_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        answers=answers,
                        rewards=rewards,
                        attempt_used=attempt_used,
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    SamplesBOX(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        sys2_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        answers=answers,
                        
                    )
                )

        if self.send_to_ref:
            response = [output.outputs[0].text for output in all_outputs]
            ray.get(self.initial_model.append.remote(prompts=all_prompts, responses=response))

        return samples_list
    
    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        sys2_mask:torch.Tensor,
        gamma: float,
        lambd: float,
        sequences:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, sys2_mask, gamma, lambd, sequences)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        if not self.strategy.args.multi_attempt:
            for t in reversed(range(response_length)):
                nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
                delta = rewards[:, t] + gamma * nextvalues - values[:, t]
                lastgaelam = delta + gamma * lambd * lastgaelam
                advantages_reversed.append(lastgaelam)            
        else:
            for t in reversed(range(response_length)):
                nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
                next_mask = sys2_mask[:, t + 1] if t < response_length - 1 else torch.zeros_like(sys2_mask[:, 0])
                gamma_mask = torch.where(next_mask, 1, gamma)   
                lambd_mask = torch.where(next_mask, 1, lambd)                
                delta = rewards[:, t] + gamma_mask * nextvalues - values[:, t]    
                lastgaelam = delta + gamma_mask * lambd_mask * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)
            
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values

        if self.strategy.args.multi_attempt:
            advantages[sys2_mask] = 0.
            returns[sys2_mask] = 0.
        
        # save_debug_data(prefix="adv", rewards=rewards, values=values, action_mask=action_mask, sys2_mask=sys2_mask, returns=returns, advantages=advantages, sequences=sequences)
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None
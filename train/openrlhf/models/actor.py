from typing import Optional, Tuple, Union

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel, AutoConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .ring_attn_utils import convert_ring_attn_params
from .utils import log_probs_from_logits, reset_position_ids, process_sequences

class Actor(nn.Module):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        resize_embeddings=-1,
        has_value_head=False,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None
            
            self.has_value_head = has_value_head
            if has_value_head:
                config = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
                config.normalize_reward = False
                config._attn_implementation = attn_implementation
                base_class = AutoModelForCausalLM._model_mapping[type(config)]
                cls_class = _get_actor_critic_model(base_class)
                self.model = cls_class.from_pretrained(
                    pretrain_or_model,
                    config=config,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if bf16 else "auto",
                    quantization_config=nf4_config,
                    device_map=device_map,
                    **kwargs,
                )
                value_head = getattr(self.model, "ac_value_head")
                if dschf is not None:
                    with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
                else:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    pretrain_or_model,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                    quantization_config=nf4_config,
                    torch_dtype=torch.bfloat16 if bf16 else "auto",
                    device_map=device_map,
                )
            if resize_embeddings > 0:
                self.model.resize_token_embeddings(resize_embeddings)

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": True,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # Call generate
        sequences = self.model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]
        # TODO: eos token generated by actor should not be masked; but those generated automatically should be masked
        raise NotImplementedError()

        return process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Returns action log probs"""
        if not self.packing_samples:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
        else:
            if ring_attn_group is not None:
                sequences, attention_mask, position_ids = convert_ring_attn_params(
                    sequences, attention_mask, packed_seq_lens, ring_attn_group
                )
            else:
                # reset the positions for packed samples
                position_ids = reset_position_ids(attention_mask)
        position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)

        if num_actions is None:
            assert return_output
            return output

        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if not self.packing_samples:
            action_log_probs = log_probs[:, -num_actions:]
        else:
            assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
            action_log_probs = []
            offset = 0
            for num_action, seq_len in zip(num_actions, packed_seq_lens):
                start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                action_log_probs.append(log_probs[:, start:end])
                offset += seq_len
            action_log_probs = torch.cat(action_log_probs, dim=1)

        if self.has_value_head and return_output:
            values = output['values'][:, :-1]
            if not self.packing_samples:
                action_values = values[:, -num_actions:]
            else:
                assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
                action_values = []
                offset = 0
                for num_action, seq_len in zip(num_actions, packed_seq_lens):
                    start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                    action_values.append(values[:, start:end])
                    offset += seq_len
                action_values = torch.cat(action_values, dim=1)  
            output["values"] = action_values      

        if return_output:
            return (action_log_probs, output)
        else:
            return action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def sleep(self):
        self.model = self.model.to("cpu")
        torch.cuda.empty_cache()

    def wake_up(self):
        self.model = self.model.to("cuda")
        torch.cuda.empty_cache()


def _get_actor_critic_model(base_llm_model):
    class ActorCriticModel(base_llm_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):            
            super().__init__(config)     
            self.ac_value_head = nn.Linear(config.hidden_size, 1, bias=False)     
            self.packing_samples = False

        def forward(self, *args, **kwargs) -> torch.Tensor:
            outputs = super().forward(*args, **kwargs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            outputs["values"] = self.ac_value_head(last_hidden_states).squeeze(-1)
            return outputs        
    return ActorCriticModel

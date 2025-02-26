import math
import logging
import os
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import ray

import torch

from openrlhf.models import Actor
from openrlhf.utils import DeepspeedStrategy
from .launcher import BasePPORole

@ray.remote(num_gpus=1)
class ReferenceModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps=None):
        self._setup_distributed(strategy)
        model = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            packing_samples=strategy.args.packing_samples,
        )
        strategy.print(model)

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()
        self.model_sleep = False

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        if self.model_sleep:
            self.model.wake_up()
            self.model_sleep = False
        with torch.no_grad():
            log_probs = self.model(
                sequences.to(device),
                num_actions,
                attention_mask.to(device),
                return_output=return_output,
                packed_seq_lens=packed_seq_lens,
            )
        return log_probs.to("cpu")

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def sleep(self) -> None:        
        self.model.sleep()
        self.model_sleep = True

import os
from typing import Dict, List, Type

import torch
import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.utils.logging_utils import init_logger
from meval import math_eval
import importlib

logger = init_logger(__name__)

class SLLMRayActor:
    def __init__(self, *args, **kwargs):
        import vllm

        self.__version__ = vllm.__version__
        assert self.__version__ >= "0.4.1", "OpenRLHF only supports vLLM >= 0.4.1"

        if "tensor_parallel_size" not in kwargs:
            kwargs["tensor_parallel_size"] = 1
        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1

        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.use_gpu_executor:
            from openrlhf.trainer.ray.vllm_worker_wrap import WorkerWrap

            vllm.worker.worker.Worker = WorkerWrap
        else:
            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            kwargs["worker_use_ray"] = True

            if vllm.__version__ > "0.4.1":
                RayWorkerWrapperPath = vllm.executor.ray_utils
            else:
                RayWorkerWrapperPath = vllm.engine.ray_utils

            class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                def __init__(self, *args, **kwargs) -> None:
                    kwargs["worker_module_name"] = "openrlhf.trainer.ray.vllm_worker_wrap"
                    kwargs["worker_class_name"] = "WorkerWrap"
                    super().__init__(*args, **kwargs)

            RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper

        print("Initializing vllm...")
        self.llm = vllm.LLM(*args, **kwargs)
        print("Initialized vllm.")
        self.llm_initialized = True
        self.llm_sleep = False
        self.sleep_enabled = kwargs.get("enable_sleep_mode", False)

    def generate(self, *args, **kwargs):
        assert self.llm_initialized
        self.wake_up()
        return self.llm.generate(*args, **kwargs)
    
    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        assert self.llm_initialized
        self.wake_up()
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address, master_port, rank_offset, world_size, group_name, backend
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend
            )

    def update_weight(self, name, dtype, shape, empty_cache=False, sleep=False):
        assert self.llm_initialized
        self.wake_up()
        self.stop_remote_worker_execution_loop()

        if self.use_gpu_executor:
            out = self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
        else:
            out = self.llm.llm_engine.model_executor._run_workers("update_weight", name, dtype, shape, empty_cache)        
        if empty_cache:
            torch.cuda.empty_cache()
        if sleep:
            self.sleep()        
        return out

    def stop_remote_worker_execution_loop(self):
        assert self.llm_initialized
        self.wake_up()
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__version__ > "0.4.2":
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()

    def eval(self, data_names, output_dir, args):
        assert self.llm_initialized
        self.wake_up()
        args = dict(
            data_names = data_names,
            data_dir = importlib.resources.files("meval").joinpath("data"),
            output_dir = output_dir,
            prompt_type = "qwen-boxed",
            custom_prompt = args.custom_sys_prompt,
            num_test_sample = -1,
            max_tokens_per_call = args.generate_max_len,
            seed = 0,
            temperature = 0,
            n_sampling = 1,
            top_p = 1,
            start = 0,
            end = -1,
            use_vllm = True,
            save_outputs = True,
            no_verbose = True,
            overwrite = True,
        )
        args = math_eval.parse_args(args)
        results = math_eval.setup(args, self.llm)
        # in the form of {data_names[0]: {'acc': x, 'response_length': x}, ...}
        return results
    
    def sleep(self):
        torch.cuda.empty_cache()
        if self.sleep_enabled:
            self.llm.sleep()
            self.llm_sleep = True

    def wake_up(self):
        if self.llm_sleep: 
            self.llm_sleep = False
            self.llm.wake_up()

    def is_ready(self):
        return self.llm_initialized
    
    def reset_prefix_cache(self):
        self.llm.reset_prefix_cache()

@ray.remote
class LLMRayActor(SLLMRayActor):
    pass

def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    pgs: List[PlacementGroup] = [],
    vllm_cls: Type[LLMRayActor] = LLMRayActor,
    **kwargs,
):  
    vllm_engines = []
    print("Init vllm args:", kwargs)
    for i in range(num_engines):
        # When tensor_parallel_size=1, vLLM init model in LLMEngine directly, assign 1 GPU for it.
        num_gpus = int(tensor_parallel_size == 1)
        scheduling_strategy = None

        if len(pgs) == 0 and tensor_parallel_size > 1:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )
        elif len(pgs) > 0:
            pg = pgs[i//(num_engines//len(pgs))]
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=i * tensor_parallel_size // num_engines,
            )
            num_gpus = 0.01

        vllm_engines.append(
            vllm_cls.options(
                num_cpus=1,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                pretrain,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                **kwargs,
            )
        )
    #ray.get([v.is_ready.remote() for v in vllm_engines])
    return vllm_engines


if __name__ == "__main__":
    llm = LLMRayActor.remote("meta-llama/Llama-2-7b-chat-hf", tensor_parallel_size=4)
    output = ray.get(llm.generate.remote("San Franciso is a"))
    print(f"output: {output}")

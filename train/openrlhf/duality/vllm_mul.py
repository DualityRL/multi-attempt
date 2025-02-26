from typing import Optional
import time
import numpy as np

from vllm.outputs import RequestOutput, CompletionOutput
from transformers import AutoTokenizer
from vllm.sampling_params import SamplingParams

import ray

from openrlhf.trainer.ppo_utils.qwen_math_eval_toolkit.grader import math_equal, math_equal_hf
from openrlhf.trainer.ray.vllm_engine import SLLMRayActor
from openrlhf.utils import tqdm, parallel_f, save_debug_data

from meval import math_eval
import importlib

from openrlhf.duality.utils import clone_sampling_params, extract_answer, filter_concat, filter_list

class RewardedRequestOutput(RequestOutput):
    """A subclass of RequestOutput that includes a reward score."""
    
    def __init__(
        self,
        reward: Optional[float] = None,
        attempt_used: Optional[float] = None,
        *args,  # Keep positional arguments for RequestOutput
        **kwargs  # Keep keyword arguments for RequestOutput
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reward = reward
        self.attempt_used = attempt_used        

    def __repr__(self) -> str:
        return super().__repr__()[:-1] + f", reward={self.reward}), attempt_used={self.attempt_used}"

@ray.remote
class MulLLM(SLLMRayActor):
    def __init__(self, 
                 *args, 
                 num_workers=16, 
                 min_attempt=1, 
                 max_attempt=5, 
                 attempt_discount=1,
                 token_per_step=64, 
                 repeat_question=False, 
                 custom_sys_prompt=None, 
                 custom_sys_prompt_wrong=None,                  
                 **kwargs):        

        kwargs["enable_prefix_caching"] = True
        self.use_hf_math = kwargs.pop("use_hf_math", False)
        super().__init__(*args, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(kwargs["model"] if "model" in kwargs else args[0], trust_remote_code=True)
        self.num_workers = num_workers
        self.mini_batch_size = 256        
        
        self.repeat_question = repeat_question

        if not custom_sys_prompt:
            if not self.repeat_question:
                self.sys_prompt = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}. You have {remaining_attempt} attempts to answer the questions, and you should provide the full reasoning steps in each attempt.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
            else:
                self.sys_prompt = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}. You have {remaining_attempt} attempts to answer the question.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
        else:
            self.sys_prompt = custom_sys_prompt

        if not custom_sys_prompt_wrong:
            if not self.repeat_question:
                self.sys_prompt_wrong = "<|im_end|>\n<|im_start|>system\nYour answer is incorrect. You have {remaining_attempt} attempts left. Try an alternative.<|im_end|>\n<|im_start|>assistant\n"
            else:
                self.sys_prompt_wrong = "<|im_end|>\n<|im_start|>system\nYour answer is incorrect. Try an alternative. You have {remaining_attempt} attempts to answer the question.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
        else:
            assert custom_sys_prompt_wrong.startswith("<|im_end|>\n<|im_start|>system")
            assert custom_sys_prompt_wrong.endswith("<|im_start|>assistant\n")
            self.sys_prompt_wrong = custom_sys_prompt_wrong

        self.token_per_step = token_per_step
        self.min_attempt = min_attempt
        self.max_attempt = max_attempt
        self.attempt_discount = attempt_discount

    def generate(self, prompt, sampling_params=None, answer=None, eval=False, use_tqdm=True):     
        if isinstance(prompt, str):
            prompt = [prompt]
        if eval:
            answer = None

        mini_batch_size = self.mini_batch_size
        prompt_batches = [prompt[i:i+mini_batch_size] for i in range(0, len(prompt), mini_batch_size)]
        if answer is not None:
            answer_batches = [answer[i:i+mini_batch_size] for i in range(0, len(answer), mini_batch_size)]

        outputs = []
        start_time = time.time()        
        if use_tqdm: pbar = tqdm(total=len(prompt), desc="Generating")           

        for n, prompt_batch in enumerate(prompt_batches):
            self.reset_prefix_cache()
            answer_batch = answer_batches[n] if answer is not None else None
            outputs.extend(self._generate(prompt_batch, sampling_params=sampling_params, answer=answer_batch))        
            if use_tqdm: pbar.update(len(prompt_batch))
        if answer is not None:
            reward = np.array([x.reward for x in outputs])
            acc = np.mean(reward > 0)
            acc_str = f" (acc: {acc*100:.2f})"
        else:
            acc_str = ""
        print(f"Complete processing {len(prompt)} prompts in {time.time() - start_time:.2f} seconds{acc_str}.")
        #save_debug_data("tmp", prefix="mllm", prompt=prompt, answer=answer, reward=[x.reward for x in outputs], response=[x.outputs[0].text for x in outputs])
        return outputs    

    def _generate(self, prompt, sampling_params=None, answer=None):
        eval = answer is None
        batch_size = len(prompt)

        if sampling_params is None:
            sampling_params = SamplingParams()

        if eval:
            remaining_attempt = np.ones((batch_size,), dtype=np.int64)
        else:
            remaining_attempt = np.random.choice(self.max_attempt + 1 - self.min_attempt, size=batch_size) + self.min_attempt 

        question = prompt

        prompt = [self.sys_prompt.format(remaining_attempt=n, input=q) for n, q in zip(remaining_attempt, question)]
        num_iteration = max((sampling_params.max_tokens // self.token_per_step) + self.max_attempt, self.max_attempt)

        sampling_params = clone_sampling_params(
            sampling_params,
            max_tokens = self.token_per_step,            
        )

        eot = [False] * batch_size  # flags for "end-of-text" reached per sample

        full_response = [x for x in prompt]
        current_response = [''] * batch_size
        reward = [-1.] * batch_size # default -1 reward
        attempt_used = [1] * batch_size

        #print("full_response[0]", full_response[0])
        for j in range(num_iteration):
            active_input = filter_list(full_response, eot)           
            start_time = time.time()
            #ray.logger.info(f"{j}/{num_iteration} Start self.llm.generate.")
            out = self.llm.generate(active_input, sampling_params, use_tqdm=False)   
            #ray.logger.info(f"{j}/{num_iteration}Complete self.llm.generate in {time.time() - start_time:.2f} seconds.")
            if j == 0: first_out = out

            active_response = [output.outputs[0].text.replace("<|im_start|>", "").replace("<|im_end|>", "") for output in out]    
            current_response = filter_concat(current_response, active_response, eot)
            full_response = filter_concat(full_response, active_response, eot)

            if not eval:                
                #ray.logger.info(f"{j}/{num_iteration} Start self.llm.check_outputs.")
                active_status = self.check_outputs(filter_list(current_response, eot), filter_list(answer, eot))                
                #ray.logger.info(f"{j}/{num_iteration} Complete self.llm.check_outputs in {time.time() - start_time:.2f} seconds.")
            else:
                active_status = [0] * batch_size

            # 0 - no box found; 1 - boxed wrong answer found; 2 - boxed correct answer found
            for active_idx, output_ in enumerate(out):
                output = output_.outputs[0]
                if output.finish_reason == "stop" and active_status[active_idx] == 0:
                    active_status[active_idx] = 3 # 3 - eos found without box
                if len(output.token_ids) == 0:
                    active_status[active_idx] = 4 # 4 - exceed max token    
            
            active_idx = 0
            new_eot = []
            for idx, finished in enumerate(eot):
                if not finished:
                    status = active_status[active_idx]
                    if status in [1, 3] and remaining_attempt[idx] > 1: 
                        # wrong boxed answer / eos output; new attempt is granted
                        remaining_attempt[idx] = remaining_attempt[idx] - 1
                        current_response[idx] = ''
                        wrong_response = self.sys_prompt_wrong.format(remaining_attempt=remaining_attempt[idx], input=question[idx])
                        full_response[idx] += wrong_response
                        attempt_used[idx] += 1
                        new_eot.append(False)
                    elif status in [1, 3] and remaining_attempt[idx] <= 1: 
                        # attempt exhausted
                        new_eot.append(True)
                    elif status == 2: 
                        # correct answer
                        if self.attempt_discount == 1:
                            reward[idx] = 1.
                        else:
                            reward[idx] = self.attempt_discount ** attempt_used[idx]
                        new_eot.append(True)
                    elif status == 4: 
                        # exceed max token
                        new_eot.append(True)
                    else:
                        # 0 status, no need to do anything
                        new_eot.append(False)

                    if status == 1: 
                        reward[idx] = -0.5 # box can be found in any attempts
                    elif status == 2: 
                        reward[idx] = +1. # correct answer
                    active_idx += 1            
                else:
                    new_eot.append(True)

            eot = new_eot        
            if all(eot): break 
        
        request_outputs = []

        for i in range(batch_size):  
            prompt_i = prompt[i]
            text_i = full_response[i][len(prompt_i):]
            completion_output = CompletionOutput(
                index=0,
                text=text_i,
                token_ids=self.tokenizer.encode(text_i, add_special_tokens=True),
                cumulative_logprob=None,
                logprobs=None,  # TODO
            )
            reequest_output = RewardedRequestOutput(
                request_id = first_out[i].request_id,
                prompt = first_out[i].prompt,
                prompt_token_ids = first_out[i].prompt_token_ids,
                prompt_logprobs=None,
                outputs=[completion_output],
                finished=finished,
                reward = reward[i],
                attempt_used = attempt_used[i],
                )
            request_outputs.append(reequest_output)

        return request_outputs
    
    def check_outputs(self, response, answer):
        # possible status - 
        # 0: no box found
        # 1: boxed wrong answer found
        # 2: boxed correct answer found
        status = [0] * len(response)
        to_check = []
        check_idx = []
        for n, x in enumerate(response):            
            pred_ans = extract_answer(x)
            if pred_ans:
                to_check.append((pred_ans, answer[n]))
                check_idx.append(n)        
        if to_check:
            #ray.logger.info(f"Start wait parallel_f.")
            f = math_equal_hf if self.use_hf_math else math_equal
            check_result = parallel_f(f, to_check, num_workers=self.num_workers)        
            #ray.logger.info(f"Finish parallel_f.")
            for n, i in enumerate(check_idx):
                status[i] = 2 if check_result[n] else 1            
        return status  
    
    def eval(self, data_names, output_dir, args, multi_attempt=False):
        assert self.llm_initialized
        self.wake_up()
        args = dict(
            data_names = data_names,
            data_dir = importlib.resources.files("meval").joinpath("data"),
            output_dir = output_dir,
            prompt_type = "none",
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
            multi_attempt = multi_attempt,
        )
        args = math_eval.parse_args(args)
        results = math_eval.setup(args, self)
        # in the form of {data_names[0]: {'acc': x, 'response_length': x}, ...}
        return results
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="HF model name or path")
    parser.add_argument("--dataset", type=str, default="olympiadbench", help="Dataset to eval")
    parser.add_argument("--min_attempt", type=int, default=5, help="Minimum attempt allowed")
    parser.add_argument("--max_attempt", type=int, default=5, help="Maximum attempt allowed")
    parser.add_argument("--output_dir", type=str, default="../../large_data/eval/test/tmp", help="Output directory")    
    args = parser.parse_args()

    llm = MulLLM.options(num_gpus=1, num_cpus=1).remote(model=args.model, min_attempt=args.min_attempt, max_attempt=args.max_attempt)
    from types import SimpleNamespace
    eval_args = SimpleNamespace(generate_max_len=8000)
    out = llm.eval.remote(args.dataset, args.output_dir, args=eval_args, multi_attempt=True)
    print(ray.get(out))

    """
    sampling_params = SamplingParams(max_tokens=3000, temperature=0.6)

    prompt = ["If $x + y = 1$ and $x - y = 3$, what is the value of $x - 2y$?",
              "If $\\sqrt{5 + x} + \\sqrt{20 - x} = 7$, what is the value of $(5 + x)(20 - x)$?",
              "Let $d_1 = a^2 + 2^a + a \cdot 2^{(a+1)/2}$ and $d_2 = a^2 + 2^a - a \cdot 2^{(a+1)/2}$. If $1 \le a \le 251$, how many integral values of $a$ are there such that $d_1 \cdot d_2$ is a multiple of $5$?"
             ]
    answer = ["4", "144", "50"]    

    output = ray.get(llm.generate.remote(prompt, sampling_params, answer=answer))
    print(output)

    for n, x in enumerate(output):
        print("response %d:" % n, x.outputs[0].text)
    """

    
    

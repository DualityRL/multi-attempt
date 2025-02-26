import os
import time
import signal
import functools
import ray
#from ray.experimental.tqdm_ray import tqdm as _tqdm
from tqdm import tqdm
import re
import subprocess
import pickle
import torch

from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer
from openrlhf.utils import DeepspeedStrategy

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer

def get_strategy(args, is_critic=False):
    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size" if not is_critic else "critic_train_batch_size"),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy

def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            data = load_from_disk(dataset)
            strategy.print(f"loaded {dataset} from disk")
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            # train will contains eval? TODO
            else:
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset

def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")   

def vllm_math_evaluate(vllm_engines, output_dir, args, enable_sleep=False):
    all_data_names = ["aime24,amc23,olympiadbench_p1", "math500", "minerva_math", "olympiadbench_p2"]
    # aime24 and amc23 are quite short and can be put together
    refs = []
    for n, data_names in enumerate(all_data_names):
        m = n % len(vllm_engines)
        refs.append(vllm_engines[m].eval.remote(data_names, output_dir, args=args))
    results_list = ray.get(refs)  # List of dictionaries
    results = {}
    # Merge all dictionaries into `results`
    for result in results_list:
        results.update(result)
    results = merge_suffix_dict(results)
    accs = {}
    for k, v in results.items():
        accs[f"eval/{k}_acc"] = v["acc"]

    # Try to compute avg_acc, set to 0 if an error occurs
    try:
        accs["eval/avg_acc"] = sum(accs.values()) / len(accs)
    except Exception:
        accs["eval/avg_acc"] = 0  # Set avg_acc to 0 in case of error

    response_lengths = {}
    for k, v in results.items():
        response_lengths[f"eval/{k}_response_length"] = v["response_length"]

    # Try to compute avg_acc, set to 0 if an error occurs
    try:
        response_lengths["eval/avg_response_length"] = sum(response_lengths.values()) / len(response_lengths)
    except Exception:
        response_lengths["eval/avg_response_length"] = 0  # Set avg_acc to 0 in case of error

    eval_results = {}
    eval_results.update(accs)
    eval_results.update(response_lengths)
    print("Evaluation results: ", eval_results)
    if enable_sleep:
        refs.extend([v.sleep.remote() for v in vllm_engines])
        ray.get(refs)
    return eval_results

def merge_suffix_dict(data):
    merged_groups = {}
    # This dictionary will be our final result.
    result = {}

    # Compile a regex that matches keys ending with _s<number>
    pattern = re.compile(r'_p\d+$')

    # Iterate over each key-value pair in the original dictionary.
    for key, value in data.items():
        if pattern.search(key):
            # Extract the base name by removing the suffix.
            base = key.rsplit("_", 1)[0]
            if base not in merged_groups:
                merged_groups[base] = {
                    "total_acc": 0.0,   # Accumulate weighted sum of accuracy.
                    "total_resp": 0.0,  # Accumulate weighted sum of response_length.
                    "num_samples": 0    # Accumulate total number of samples.
                }
            n = value["num_samples"]
            # Convert response_length to float (or int) for numerical operations.
            resp = float(value["response_length"])
            merged_groups[base]["total_acc"] += value["acc"] * n
            merged_groups[base]["total_resp"] += resp * n
            merged_groups[base]["num_samples"] += n
        else:
            # If key doesn't match the suffix pattern, keep it unchanged.
            result[key] = value

    # Compute weighted averages for each merged group.
    for base, stats in merged_groups.items():
        n = stats["num_samples"]
        weighted_acc = stats["total_acc"] / n if n > 0 else 0
        weighted_resp = stats["total_resp"] / n if n > 0 else 0
        result[base] = {
            "acc": weighted_acc,
            "num_samples": n,
            "response_length": weighted_resp
        }
    return result

def get_git_version():
    """Returns the current Git commit hash or branch name."""
    try:
        # Get the current commit hash
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).strip().decode()
        # Get the current branch/tag name
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).strip().decode()
        return f"{branch_name} ({commit_hash})"
    except subprocess.CalledProcessError:
        return "Git version unavailable"

"""
class tqdm:
    def __init__(self, *args, disable=False, **kwargs):
        self.disabled = disable
        self.postfix = {}  # Store postfix values
        self.description = kwargs.get("desc", "")  # Store initial description

        if not self.disabled:
            self._tqdm = _tqdm(*args, **kwargs)  # Create actual tqdm instance
        else:
            self._tqdm = None  # No tqdm instance when disabled

    def update(self, n=1):
        if self._tqdm:
            self._tqdm.update(n)

    def close(self):
        if self._tqdm:
            self._tqdm.close()

    def __iter__(self):
        if self._tqdm:
            return iter(self._tqdm)
        return iter([])  # Return an empty iterator when disabled

    def __enter__(self):
        if self._tqdm:
            return self._tqdm.__enter__()
        return self  # Return self to prevent errors in `with tqdm(...) as pbar:`

    def __exit__(self, exc_type, exc_value, traceback):
        if self._tqdm:
            self._tqdm.__exit__(exc_type, exc_value, traceback)

    def set_description(self, desc, refresh=True):
        self.description = desc  # Store description text
        if self._tqdm:
            self._tqdm.set_description(desc)
            if refresh:
                self._tqdm.refresh()

    def set_postfix(self, kwargs):
        self.old_description = self.description
        postfix_str = " | " + " ".join(f"{k}={v:.3f}" for k, v in kwargs.items()) + " | "
        self.set_description(self.description + postfix_str)
        self.description = self.old_description
"""

def save_debug_data(directory="large_data/tmp", prefix="debug", **kwargs):
    """
    Save arbitrary keyword arguments as a dictionary in a unique pickle file.

    Args:
        directory (str): Directory where the file will be saved.
        prefix (str): Filename prefix (default is "debug").
        **kwargs: Any keyword arguments to save.

    Returns:
        str: The filename where the data was saved.
    """
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

    # Find the smallest available filename
    n = 0
    while os.path.exists(f"{directory}/{prefix}_{n}.pickle"):
        n += 1
    filename = f"{directory}/{prefix}_{n}.pickle"

    # Process data: If a value is a tensor, detach & move to CPU
    processed_data = {
        key: (value.detach().cpu() if isinstance(value, torch.Tensor) else value)
        for key, value in kwargs.items()
    }

    # Save the processed dictionary
    with open(filename, "wb") as f:
        pickle.dump(processed_data, f)

    print(f"Saved debug data to {filename}")
    return filename  # Return the saved filename            

def load_debug_data(directory="large_data/tmp", prefix="debug", max_n=-1):
    merged_data = {}
    # Get all matching files
    files = sorted([f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".pickle")])
    n = 0

    for file in files:
        file_path = os.path.join(directory, file)        
        # Load the pickle file
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # Merge into the large dictionary
        for key, value in data.items():
            if key in merged_data:
                merged_data[key].append(value)  # Append non-tensors
            else:
                merged_data[key] = [value]
        n += 1
        if n >= max_n and max_n > 0:
            break
    print(f"Loaded {len(files)} files from {directory} with prefix '{prefix}'")
    return merged_data

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def timeout_decorator(seconds, default_value):
    """
    A decorator that raises a TimeoutException (and returns default_value)
    if the decorated function runs longer than 'seconds'.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set the alarm signal handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            except TimeoutException:
                print("timeout occured with args:", args, kwargs)
                return default_value
            finally:
                signal.alarm(0)  # Disable the alarm
        return wrapper
    return decorator

@ray.remote
def execute_task(f, arg, timeout, default_value):
    """
    Wraps the function f with a timeout. If execution exceeds timeout seconds,
    returns default_value. Supports passing a dict (keyword args), tuple (positional args),
    or a single value.
    """
    # Wrap f with the timeout logic
    f_with_timeout = timeout_decorator(timeout, default_value)(f)
    if isinstance(arg, dict):
        return f_with_timeout(**arg)
    elif isinstance(arg, tuple):
        return f_with_timeout(*arg)
    else:
        return f_with_timeout(arg)

def parallel_f(f, args_list, num_workers=32, timeout=10, default_value=False):
    """
    Runs function f on each element of args_list via Ray remote tasks.
    No more than num_workers tasks will be running concurrently.
    Each task applies an internal timeout (in seconds). If a task exceeds
    the timeout, it is cancelled and default_value is used.
    
    Additionally, the elapsed time for each task is recorded.
    
    Returns:
        results: a list of the function results (or default_value if timed out)
        times: a list of elapsed times for each task
    """
    results = [None] * len(args_list)
    pending_tasks = list(enumerate(args_list))  # Each element: (index, argument)
    # running_tasks maps future -> (index, start_time)
    running_tasks = {}
    timeout_tasks = {}

    while pending_tasks or running_tasks:
        # Launch new tasks while we haven't hit the concurrency limit.
        while len(running_tasks) < num_workers and pending_tasks:
            idx, arg = pending_tasks.pop(0)
            future = execute_task.remote(f, arg, timeout, default_value)
            running_tasks[future] = (idx, time.time())

        current_time = time.time()

        to_cancel = []
        for future, (idx, start_time) in running_tasks.items():
            elapsed = current_time - start_time
            if elapsed > timeout:
                if idx not in timeout_tasks:
                    timeout_tasks[idx] = current_time
                    ray.logger.info(f"Timeout occured with idx {idx} ({elapsed:.1f}s) args {args_list[idx]}")
                elif current_time - timeout_tasks[idx] > timeout:
                    timeout_tasks[idx] = current_time
                    ray.logger.info(f"Continue timeout occured with idx {idx} ({elapsed:.1f}s) args {args_list[idx]}")
            
            if elapsed > 180:            
                to_cancel.append(future)
                ray.logger.info(f"Forceful termination with idx {idx} ({elapsed:.1f}s) args {args_list[idx]}")

        for future in to_cancel:
            idx, start_time = running_tasks.pop(future)
            # Cancel the task (forcefully, if necessary), wrapped in a try/except
            try:
                ray.cancel(future, force=True)
            except Exception:
                pass
            results[idx] = default_value

        # Poll for finished tasks.
        if running_tasks:
            done, _ = ray.wait(list(running_tasks.keys()), num_returns=len(running_tasks), timeout=0.1)
            for future in done:
                # Check if it wasn't already cancelled.
                if future in running_tasks:
                    idx, start_time = running_tasks.pop(future)
                    try:
                        results[idx] = ray.get(future)
                    except Exception:
                        results[idx] = default_value                    

        # Sleep briefly to avoid busy-waiting.
        time.sleep(0.01)

    return results
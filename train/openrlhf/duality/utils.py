from vllm import SamplingParams

def clone_sampling_params(sampling_params, **kwargs):
    # For some reasons, simply cloning sample_params then modify the property will not work
    internal_fields = {"_real_n", "output_text_buffer_length", "_all_stop_token_ids"}
    params_dict = {
        field: getattr(sampling_params, field)
        for field in sampling_params.__annotations__
        if field not in internal_fields and not field.startswith("_")
    }
    params_dict.update(kwargs)
    return SamplingParams.from_optional(**params_dict)

def extract_answer(pred_str):
    if "boxed{" not in pred_str:
        return ""
    ans = pred_str.split("boxed{")[-1]
    stack = 1
    a = ""
    for c in ans:
        if c == "{":
            stack += 1
            a += c
        elif c == "}":
            stack -= 1
            if stack == 0:
                return a
            a += c
        else:
            a += c
    return ""

def filter_concat(x, y, eot):
    # only concat y[i] to x[j] if not eot[j]
    i = 0
    for j, finished in enumerate(eot):
        if not finished:
            x[j] = x[j] + y[i]
            i += 1
    return x

def filter_list(x, eot):
    return [inp for inp, finished in zip(x, eot) if not finished]


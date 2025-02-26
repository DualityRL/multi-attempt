import torch
import torch.distributed as dist
import torch.nn.functional as F
import re

def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


def exist_and_not_none(d, key):
    return key in d and not d[key] is None

def create_token_mask(input_tokens, tokenizer, start_token="<|intuition_start|>", end_token="<|intuition_end|>"):
    """
    Creates a mask where tokens between and including `start_token` and `end_token` are set to 1, else 0.
    Supports `start_token` and `end_token` that tokenize into multiple token IDs.

    Args:
        input_tokens (List[int]): Tokenized input (list of token IDs).
        tokenizer: The tokenizer used for tokenizing the input.
        start_token (str): Special start token that may tokenize into multiple token IDs.
        end_token (str): Special end token that may tokenize into multiple token IDs.

    Returns:
        List[int]: A binary mask of the same length as input_tokens.
    """
    mask = [0] * len(input_tokens)
    inside_special = False  # Flag to track masking state

    # Tokenize the start and end tokens to get their corresponding token ID sequences
    start_token_ids = tokenizer.encode(start_token, add_special_tokens=False)
    end_token_ids = tokenizer.encode(end_token, add_special_tokens=False)

    i = 0
    while i < len(input_tokens):
        # Check for the start token sequence
        if input_tokens[i:i+len(start_token_ids)] == start_token_ids:
            inside_special = True
            mask[i:i+len(start_token_ids)] = [1] * len(start_token_ids)
            i += len(start_token_ids)
            if i >= len(input_tokens):
                break  # Exit the loop if we've reached the end

        # Apply mask while inside the special region
        if inside_special:
            mask[i] = 1
            # Check for the end token sequence
            if input_tokens[i:i+len(end_token_ids)] == end_token_ids:
                inside_special = False
                mask[i:i+len(end_token_ids)] = [1] * len(end_token_ids)
                i += len(end_token_ids) - 1  # Move pointer to the end of end_token_ids
        
        i += 1  # Move to the next token
    return mask

def remove_intuition_tags(text: str) -> str:
    """
    Removes all occurrences of the pair tags <|intuition_start|> and <|intuition_end|>
    along with any content in between.

    Args:
        text (str): The input string possibly containing one or more occurrences
                    of the pair tags.

    Returns:
        str: The string with the specified tags and content between them removed.
    """
    # The regex pattern uses non-greedy matching (.*?)
    pattern = r'<\|intuition_start\|>.*?<\|intuition_end\|>'
    # re.DOTALL ensures that the dot (.) matches newline characters as well
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text
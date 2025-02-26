#!/usr/bin/env python3
"""
Script to convert a DeepSpeed ZeRO checkpoint directly to a Hugging Face model.

This script performs the following steps:
  1. Loads a ZeRO checkpoint from the provided checkpoint directory (using a function from zero_to_fp32.py).
  2. Loads a Hugging Face configuration (and tokenizer config) from the provided path.
  3. Instantiates a Hugging Face model (using AutoModelForCausalLM; modify as needed).
  4. Loads the consolidated FP32 state dict into the model.
  5. Saves the model in Hugging Face format to an output subdirectory inside the checkpoint folder.
     The output subdirectory is named `<tag>_hf` (for example, global_step12_hf) if a tag is provided,
     or `latest_hf` otherwise.
  6. Loads the tokenizer from the same path (assumed to contain the tokenizer config) and saves it into the output directory.

Usage:
    python convert_zero_to_hf.py \
         <checkpoint_dir> \
         <path_to_config> \
         [-t <tag>] \
         [--exclude_frozen_parameters] [-d|--debug]
"""

import argparse
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Import the zero_to_fp32 module.
# Ensure that zero_to_fp32.py is in your PYTHONPATH or in the same directory.
import zero_to_fp32


def main():
    parser = argparse.ArgumentParser(
        description="Convert a DeepSpeed ZeRO checkpoint directly to a Hugging Face model."
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to the desired checkpoint folder (e.g., /path/checkpoint-12)"
    )
    parser.add_argument(
        "path_to_config",
        type=str,
        help="Path to the Hugging Face config file or directory (should contain config.json and tokenizer config)"
    )
    parser.add_argument(
        "-t", "--tag",
        type=str,
        default=None,
        help="Checkpoint tag used as a unique identifier for the checkpoint, e.g., global_step12"
    )
    parser.add_argument(
        "--exclude_frozen_parameters",
        action='store_true',
        help="Exclude frozen parameters"
    )
    parser.add_argument(
        "-d", "--debug",
        action='store_true',
        help="Enable debug mode"
    )
    args = parser.parse_args()

    # Pass the debug flag to the zero_to_fp32 module.
    zero_to_fp32.debug = args.debug

    # --- Step 1. Extract the consolidated FP32 state dict from the ZeRO checkpoint ---
    print("Extracting FP32 state dict from ZeRO checkpoint...")
    state_dict = zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
        args.checkpoint_dir,
        tag=args.tag,
        exclude_frozen_parameters=args.exclude_frozen_parameters
    )

    # --- Step 2. Load the Hugging Face configuration ---
    print(f"Loading Hugging Face config from {args.path_to_config} ...")
    config = AutoConfig.from_pretrained(args.path_to_config)

    # --- Step 3. Instantiate the model ---
    # Change AutoModelForCausalLM to another model class if needed.
    print("Instantiating model from config...")
    model = AutoModelForCausalLM.from_config(config)

    # --- Step 4. Load the state dict into the model ---
    print("Loading state dict into model...")
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print("Warning: missing keys:", load_result.missing_keys)
    if load_result.unexpected_keys:
        print("Warning: unexpected keys:", load_result.unexpected_keys)

    # --- Step 5. Save the model in Hugging Face format ---
    # The output directory will be a subdirectory of the checkpoint directory.
    tag_name = args.tag if args.tag is not None else "latest"
    output_dir = os.path.join(os.path.abspath(args.checkpoint_dir) + "_hf", tag_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving Hugging Face model to {output_dir} ...")
    model.save_pretrained(output_dir)

    # --- Step 6. Load and save the tokenizer ---
    # We assume the tokenizer configuration is in the same directory as the model config.
    print(f"Loading tokenizer from {args.path_to_config} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.path_to_config)
    print(f"Saving tokenizer to {output_dir} ...")
    tokenizer.save_pretrained(output_dir)

    print("Conversion complete. Your Hugging Face model is saved at:")
    print(output_dir)


if __name__ == "__main__":
    main()

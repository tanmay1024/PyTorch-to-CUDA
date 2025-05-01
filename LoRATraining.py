#!/usr/bin/env python
# coding=utf-8

import os
import sys
import json
import argparse
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, concatenate_datasets, DatasetDict
from DifficultyProgression import DifficultyProgressionCallback, StratifiedSamplingDataset
# Hugging Face imports
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint
from TrainingArgs import ModelArguments, DataArguments, DifficultyProgressionArguments # Import new dataclasses

# Custom metrics
import re
from rouge_score import rouge_scorer
import Levenshtein

# Set up logging
logger = logging.getLogger(__name__)


def preprocess_function(examples, tokenizer, data_config, training_config):
    """
    Preprocessing function to tokenize PyTorch and CUDA code examples.
    """
    # Format inputs and outputs with special tokens
    pytorch_codes = examples[data_config.pytorch_column]
    cuda_codes = examples[data_config.cuda_column]
    
    # Format with special tokens
    formatted_inputs = [f"<pytorch>{code}</pytorch>" for code in pytorch_codes]
    formatted_outputs = [f"<cuda>{code}</cuda>" for code in cuda_codes]
    
    # For training, we need the combined format
    combined_texts = []
    for pytorch, cuda in zip(formatted_inputs, formatted_outputs):
        combined_text = f"{pytorch}\n{cuda}"
        combined_texts.append(combined_text)
    
    # Tokenize
    tokenized_inputs = tokenizer(
        combined_texts,
        max_length=data_config.max_pytorch_length + data_config.max_cuda_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Prepare labels for language modeling (same as input_ids)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    # If using contrastive learning and we have correct/incorrect labels
    if data_config.use_contrastive and data_config.is_correct_column in examples:
        is_correct = examples[data_config.is_correct_column]
        tokenized_inputs["is_correct"] = is_correct
    
    # If stratified sampling by difficulty
    if data_config.stratified_sampling and data_config.difficulty_column in examples:
        tokenized_inputs["difficulty"] = examples[data_config.difficulty_column]
    
    return tokenized_inputs

def compute_metrics(eval_preds, tokenizer, data_config):
    """
    Compute metrics for code translation evaluation.
    """
    preds, labels = eval_preds
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Extract CUDA code from predictions and labels
    cuda_preds = []
    cuda_labels = []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        # Extract CUDA code using regex
        pred_match = re.search(r"<cuda>(.*?)</cuda>", pred, re.DOTALL)
        label_match = re.search(r"<cuda>(.*?)</cuda>", label, re.DOTALL)
        
        if pred_match and label_match:
            cuda_preds.append(pred_match.group(1).strip())
            cuda_labels.append(label_match.group(1).strip())
        else:
            # If CUDA code couldn't be extracted, use the entire string
            cuda_preds.append(pred)
            cuda_labels.append(label)
    
    # Initialize metrics
    metrics = {}
    
    # ROUGE score for text similarity
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge_scores = [scorer.score(pred, label) for pred, label in zip(cuda_preds, cuda_labels)]
    
    metrics["rouge1"] = np.mean([score["rouge1"].fmeasure for score in rouge_scores])
    metrics["rouge2"] = np.mean([score["rouge2"].fmeasure for score in rouge_scores])
    metrics["rougeL"] = np.mean([score["rougeL"].fmeasure for score in rouge_scores])
    
    # Exact match (percentage of perfect translations)
    exact_matches = [pred.strip() == label.strip() for pred, label in zip(cuda_preds, cuda_labels)]
    metrics["exact_match"] = np.mean(exact_matches) * 100
    
    # Code edit distance (normalized)
    edit_distances = [Levenshtein.distance(pred, label) for pred, label in zip(cuda_preds, cuda_labels)]
    max_lengths = [max(len(pred), len(label)) for pred, label in zip(cuda_preds, cuda_labels)]
    normalized_distances = [dist / max_len if max_len > 0 else 0 for dist, max_len in zip(edit_distances, max_lengths)]
    metrics["edit_distance"] = np.mean(edit_distances)
    metrics["normalized_edit_distance"] = np.mean(normalized_distances)
    
    # Code structure similarity (based on function signatures and structure)
    # This is a simplified version - a more robust implementation would use AST parsing
    def extract_function_signatures(code):
        signatures = re.findall(r"__global__\s+void\s+\w+\([^)]*\)", code)
        return signatures
    
    pred_signatures = [extract_function_signatures(pred) for pred in cuda_preds]
    label_signatures = [extract_function_signatures(label) for label in cuda_labels]
    
    signature_matches = []
    for pred_sigs, label_sigs in zip(pred_signatures, label_signatures):
        if not label_sigs:  # No signatures in reference
            signature_matches.append(0.0)
            continue
            
        if not pred_sigs:  # No signatures in prediction
            signature_matches.append(0.0)
            continue
            
        # Count matches
        matches = sum(1 for ps in pred_sigs if any(ps == ls for ls in label_sigs))
        signature_matches.append(matches / max(len(pred_sigs), len(label_sigs)))
    
    metrics["signature_match"] = np.mean(signature_matches) * 100
    
    return metrics


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, DifficultyProgressionArguments))

    # If args are provided via config file using HfArgumentParser format:
    # model_args, data_args, training_args, difficulty_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # Or parse from command line:
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument path to json file, load from it
        logger.info(f"Loading arguments from config file: {sys.argv[1]}")
        model_args, data_args, training_args, difficulty_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        logger.info("Parsing arguments from command line.")
        model_args, data_args, training_args, difficulty_args = parser.parse_args_into_dataclasses()

    # Setup logging level based on training args
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        + f" 4-bits training: {model_args.use_4bit}, 8-bits training: {model_args.use_8bit}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Difficulty progression parameters {difficulty_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    original_dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir
    )
    all_datasets = list(original_dataset.values())
    concatenated_dataset = concatenate_datasets(all_datasets)
    train_test_dataset = concatenated_dataset.train_test_split(test_size=0.3, seed=42)

    # Further split the test into test and validation
    test_valid_dataset = train_test_dataset["test"].train_test_split(test_size=0.33, seed=42)

    # Create the final DatasetDict with all three splits
    dataset = DatasetDict({
        "train": train_test_dataset["train"],
        "test": test_valid_dataset["train"], 
        "validation": test_valid_dataset["test"]
    })

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    
    # Add special tokens
    special_tokens = {"additional_special_tokens": ["<pytorch>", "</pytorch>", "<cuda>", "</cuda>"]}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        device_map="auto",
    )
    
    # Resize token embeddings to accommodate special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA
    # target_modules = [name.strip() for name in model_args.target_modules]
    lora_config = LoraConfig(
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=model_args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Process the datasets
    preprocess_function_wrapped = lambda examples: preprocess_function(
        examples, tokenizer, data_args, training_args
    )
    
    processed_datasets = dataset.map(
        preprocess_function_wrapped,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing dataset",
    )
    
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    
    logger.info("Setting up evaluation metrics")
    compute_metrics_wrapped = lambda eval_preds: compute_metrics(
        eval_preds, tokenizer, data_args
    )

    # Create custom dataset for stratified sampling
    if difficulty_args.enable and data_args.difficulty_column in dataset["train"].column_names:
        # Define initial difficulty weights
        difficulty_weights = {
            "1": difficulty_args.initial_weights.easy,
            "2": difficulty_args.initial_weights.medium,
            "3": difficulty_args.initial_weights.hard
        }
        
        # Define target weights for the end of training
        final_weights = {
            "1": difficulty_args.final_weights.easy,
            "2": difficulty_args.final_weights.medium,
            "3": difficulty_args.final_weights.hard
        }
        
        # Create the stratified dataset
        train_dataset = StratifiedSamplingDataset(
            train_dataset, data_args.difficulty_column, difficulty_weights
        )
        
        # Create the progression callback
        difficulty_progression_callback = DifficultyProgressionCallback(
            train_dataset=train_dataset,
            initial_weights=difficulty_weights,
            final_weights=final_weights,
            progression_type="linear",  # Smooth transition
            warmup_epochs=1.0,           # 1 epoch of warmup
            log_every_epoch=1            # Log every epoch
        )
    
    # Initialize callbacks
    callbacks = []
    
    # Add difficulty progression callback if applicable
    if data_args.stratified_sampling and data_args.difficulty_column in dataset["train"].column_names:
        callbacks.append(difficulty_progression_callback)
    
    # # Add difficulty-aware evaluation callback if applicable
    # if data_config.difficulty_column in dataset["train"].column_names and eval_dataset is not None:
    #     callbacks.append(
    #         DifficultyProgressionCallback(eval_dataset, tokenizer, data_config)
    #     )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_wrapped if training_args.do_eval else None,
        callbacks=callbacks,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint := get_last_checkpoint(training_args.output_dir):
            checkpoint = last_checkpoint
            
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
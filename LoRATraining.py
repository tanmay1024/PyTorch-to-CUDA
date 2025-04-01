#!/usr/bin/env python
# coding=utf-8

import os
import torch
import logging
import numpy as np
import evaluate
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset, DatasetDict

# Hugging Face imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

# Custom metrics
import re
from rouge_score import rouge_scorer
import Levenshtein

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments for the model configuration.
    """
    model_name_or_path: str = field(
        default="codellama/CodeLlama-13b-hf",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_8bit: bool = field(
        default=True,
        metadata={"help": "Use 8-bit quantization to reduce memory usage"}
    )
    use_4bit: bool = field(
        default=False, 
        metadata={"help": "Use 4-bit quantization (QLoRA)"}
    )
    lora_rank: int = field(
        default=16,
        metadata={"help": "Rank parameter for LoRA adaptation"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Alpha parameter for LoRA adaptation"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
        metadata={"help": "Comma-separated list of module names to apply LoRA to"}
    )

@dataclass
class DataArguments:
    """
    Arguments for dataset configuration.
    """
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use"}
    )
    pytorch_column: str = field(
        default="pytorch_code",
        metadata={"help": "Column in the dataset that contains PyTorch code"}
    )
    cuda_column: str = field(
        default="cuda_code",
        metadata={"help": "Column in the dataset that contains CUDA code"}
    )
    difficulty_column: Optional[str] = field(
        default="difficulty",
        metadata={"help": "Column that indicates the difficulty level (if available)"}
    )
    is_correct_column: Optional[str] = field(
        default="is_correct",
        metadata={"help": "Column that indicates if this is a correct translation (for contrastive examples)"}
    )
    max_pytorch_length: int = field(
        default=1024,
        metadata={"help": "Maximum length for PyTorch code"}
    )
    max_cuda_length: int = field(
        default=1536,
        metadata={"help": "Maximum length for CUDA code (usually longer than PyTorch)"}
    )
    contrastive_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Weight of contrastive loss component (if using contrastive learning)"}
    )
    use_contrastive: bool = field(
        default=False,
        metadata={"help": "Whether to use contrastive learning with correct/incorrect examples"}
    )
    stratified_sampling: bool = field(
        default=True,
        metadata={"help": "Whether to use stratified sampling based on difficulty levels"}
    )
    easy_weight: float = field(
        default=0.2,
        metadata={"help": "Sampling weight for easy examples if stratified sampling is enabled"}
    )
    medium_weight: float = field(
        default=0.3,
        metadata={"help": "Sampling weight for medium examples if stratified sampling is enabled"}
    )
    hard_weight: float = field(
        default=0.5,
        metadata={"help": "Sampling weight for hard examples if stratified sampling is enabled"}
    )

# Custom callbacks
class DifficultyAwareCallback(TrainerCallback):
    """
    Callback to track metrics by difficulty level
    """
    def __init__(self, eval_dataset, tokenizer, data_args):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.data_args.difficulty_column is None:
            return
        
        # Get model from trainer
        model = kwargs.get("model", None)
        if model is None:
            return
            
        # Get metrics by difficulty
        difficulty_levels = set(self.eval_dataset[self.data_args.difficulty_column])
        
        for difficulty in difficulty_levels:
            difficulty_dataset = self.eval_dataset.filter(
                lambda example: example[self.data_args.difficulty_column] == difficulty
            )
            
            # Here you would compute metrics on this subset
            # For simplicity, we're just logging the number of examples
            logger.info(f"Difficulty {difficulty}: {len(difficulty_dataset)} examples")

def preprocess_function(examples, tokenizer, data_args, training_args):
    """
    Preprocessing function to tokenize PyTorch and CUDA code examples.
    """
    # Format inputs and outputs with special tokens
    pytorch_codes = examples[data_args.pytorch_column]
    cuda_codes = examples[data_args.cuda_column]
    
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
        max_length=data_args.max_pytorch_length + data_args.max_cuda_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Prepare labels for language modeling (same as input_ids)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    # If using contrastive learning and we have correct/incorrect labels
    if data_args.use_contrastive and data_args.is_correct_column in examples:
        is_correct = examples[data_args.is_correct_column]
        tokenized_inputs["is_correct"] = is_correct
    
    # If stratified sampling by difficulty
    if data_args.stratified_sampling and data_args.difficulty_column in examples:
        tokenized_inputs["difficulty"] = examples[data_args.difficulty_column]
    
    return tokenized_inputs

def compute_metrics(eval_preds, tokenizer, data_args):
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

class StratifiedSamplingDataset(torch.utils.data.Dataset):
    """
    Custom dataset that performs stratified sampling based on difficulty levels.
    """
    def __init__(self, dataset, difficulty_column, weights):
        self.dataset = dataset
        self.difficulty_column = difficulty_column
        self.weights = weights
        
        # Group examples by difficulty
        self.difficulty_indices = {}
        for i, example in enumerate(self.dataset):
            difficulty = example[self.difficulty_column]
            if difficulty not in self.difficulty_indices:
                self.difficulty_indices[difficulty] = []
            self.difficulty_indices[difficulty].append(i)
        
        # Calculate probabilities for each group
        total_weight = sum(weights.values())
        self.probs = {
            diff: weights.get(diff, 1.0) / total_weight 
            for diff in self.difficulty_indices.keys()
        }
        
        # Calculate number of examples per epoch
        self.num_examples = len(self.dataset)
        
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        # Sample difficulty level based on weights
        difficulties = list(self.difficulty_indices.keys())
        probs = [self.probs[d] for d in difficulties]
        difficulty = np.random.choice(difficulties, p=probs)
        
        # Sample random example from that difficulty
        indices = self.difficulty_indices[difficulty]
        sample_idx = indices[np.random.randint(0, len(indices))]
        
        return self.dataset[sample_idx]

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Log arguments
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Training arguments: {training_args}")
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Load dataset
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=training_args.cache_dir
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        use_fast=True,
    )
    
    # Add special tokens
    special_tokens = {"additional_special_tokens": ["<pytorch>", "</pytorch>", "<cuda>", "</cuda>"]}
    tokenizer.add_special_tokens(special_tokens)
    
    # Prepare quantization config
    if model_args.use_4bit:
        compute_dtype = torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif model_args.use_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    else:
        quant_config = None
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        quantization_config=quant_config,
        device_map="auto",
    )
    
    # Resize token embeddings to accommodate special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare model for k-bit training if using quantization
    if model_args.use_8bit or model_args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    target_modules = [name.strip() for name in model_args.target_modules.split(",")]
    lora_config = LoraConfig(
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
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
    eval_dataset = processed_datasets["validation"] if "validation" in processed_datasets else None
    
    # Create custom dataset for stratified sampling
    if data_args.stratified_sampling and data_args.difficulty_column in dataset["train"].column_names:
        difficulty_weights = {
            "easy": data_args.easy_weight,
            "medium": data_args.medium_weight,
            "hard": data_args.hard_weight,
        }
        train_dataset = StratifiedSamplingDataset(
            train_dataset, "difficulty", difficulty_weights
        )
    
    # Custom metrics computation
    def compute_metrics_wrapped(eval_preds):
        return compute_metrics(eval_preds, tokenizer, data_args)
    
    # Initialize callbacks
    callbacks = []
    if data_args.difficulty_column in dataset["train"].column_names and eval_dataset is not None:
        callbacks.append(
            DifficultyAwareCallback(eval_dataset, tokenizer, data_args)
        )
    
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
"""
Author: Tanmay Thakur
File: PreProcessing.py
Created: 2025-03-07 03:10
Description: 
"""
from transformers import AutoTokenizer
import torch
print(torch.backends.mps.is_available())  # Should print True
print(torch.backends.mps.is_built())      # Should print True


# Load the CodeLlama tokenizer
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

# Add special tokens to the tokenizer
special_tokens = {
    "additional_special_tokens": [
        "<pytorch>", "</pytorch>", 
        "<cuda>", "</cuda>"
    ]
}
tokenizer.add_special_tokens(special_tokens)

def preprocess_function(examples):
    # Format each example with the special tokens
    formatted_inputs = []
    formatted_outputs = []
    
    for pytorch_code, cuda_code in zip(examples["pytorch_code"], examples["cuda_code"]):
        # Format input with PyTorch code
        formatted_input = f"<pytorch>{pytorch_code}</pytorch>"
        formatted_inputs.append(formatted_input)
        
        # Format output with CUDA code
        formatted_output = f"<cuda>{cuda_code}</cuda>"
        formatted_outputs.append(formatted_output)
    
    # Tokenize the formatted inputs
    inputs = tokenizer(
        formatted_inputs,
        padding="max_length",
        truncation=True,
        max_length=1024,  # Adjust based on your code's typical length
        return_tensors="pt"
    )
    
    # Tokenize the formatted outputs
    outputs = tokenizer(
        formatted_outputs,
        padding="max_length",
        truncation=True,
        max_length=1024,  # Adjust based on your code's typical length
        return_tensors="pt"
    )
    
    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": outputs.input_ids
    }

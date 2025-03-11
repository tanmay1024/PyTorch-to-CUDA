import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

dataset = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive", cache_dir="./Datasets")

# Index(['Op_Name', 'Level_ID', 'Task_ID', 'Kernel_Name', 'CUDA_Runtime',
#        'PyTorch_Native_Runtime', 'PyTorch_Compile_Runtime',
#        'CUDA_Speedup_Native', 'CUDA_Speedup_Compile', 'CUDA_Code',
#        'PyTorch_Code_Module', 'PyTorch_Code_Functional', 'Correct', 'Max_Diff',
#        'Error', 'NCU_Profile', 'Torch_Profile', 'Clang_Tidy',
#        '__index_level_0__'],
#       dtype='object')
df_l1 = dataset["level_1"].to_pandas()
df_l2 = dataset["level_2"].to_pandas()
df_l3 = dataset["level_3"].to_pandas()
print(df_l1.shape, df_l2.shape, df_l3.shape)
# (12157, 19) (12938, 19) (5520, 19)

# Group by correct and count
print(df_l1["Correct"].value_counts())
print(df_l2["Correct"].value_counts())
print(df_l3["Correct"].value_counts())
# Load the CodeLlama tokenizer
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")

# Add special tokens to the tokenizer
special_tokens = {
    "additional_special_tokens": [
        "<pytorch>", "</pytorch>", 
        "<cuda>", "</cuda>"
    ]
}
tokenizer.add_special_tokens(special_tokens)

# Analyze the token lengths of your dataset
df_l1 = df_l1[df_l1["Correct"] == True]
df_l2 = df_l2[df_l2["Correct"] == True]
df_l3 = df_l3[df_l3["Correct"] == True]
pytorch_lengths = []
cuda_lengths = []

for name, dataset in {'Level 1': df_l1, 'Level 2': df_l2, 'Level 3': df_l3}.items():  
    for i in tqdm(range(len(dataset))):
        example = dataset.iloc[i]
        # Format with your special tokens
        pytorch_formatted = f"<pytorch>{example['PyTorch_Code_Functional']}</pytorch>"
        cuda_formatted = f"<cuda>{example['CUDA_Code']}</cuda>"
        
        # Tokenize without padding or truncation to get true lengths
        pytorch_tokens = tokenizer(pytorch_formatted, truncation=False, padding=False)
        cuda_tokens = tokenizer(cuda_formatted, truncation=False, padding=False)
        
        pytorch_lengths.append(len(pytorch_tokens["input_ids"]))
        cuda_lengths.append(len(cuda_tokens["input_ids"]))

    print(f"Dataset: {name}")       
    # Calculate statistics
    print(f"PyTorch code statistics:")
    print(f"  Mean length: {np.mean(pytorch_lengths):.1f} tokens")
    print(f"  Median length: {np.median(pytorch_lengths):.1f} tokens")
    print(f"  95th percentile: {np.percentile(pytorch_lengths, 95):.1f} tokens")
    print(f"  Max length: {max(pytorch_lengths)} tokens")

    print(f"\nCUDA code statistics:")
    print(f"  Mean length: {np.mean(cuda_lengths):.1f} tokens")
    print(f"  Median length: {np.median(cuda_lengths):.1f} tokens")
    print(f"  95th percentile: {np.percentile(cuda_lengths, 95):.1f} tokens")
    print(f"  Max length: {max(cuda_lengths)} tokens")

    print(f"\nCombined (PyTorch + CUDA) statistics:")
    combined_lengths = [p + c for p, c in zip(pytorch_lengths, cuda_lengths)]
    print(f"  Mean length: {np.mean(combined_lengths):.1f} tokens")
    print(f"  Median length: {np.median(combined_lengths):.1f} tokens")
    print(f"  95th percentile: {np.percentile(combined_lengths, 95):.1f} tokens")
    print(f"  Max length: {max(combined_lengths)} tokens")
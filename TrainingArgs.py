# training_args.py

import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="codellama/CodeLlama-13b-hf",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default="./cache",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_8bit: bool = field(
        default=False, # Defaulting to False as 4bit is often preferred now
        metadata={"help": "Use 8-bit quantization via bitsandbytes"}
    )
    use_4bit: bool = field(
        default=False, # Defaulting QLoRA to True
        metadata={"help": "Use 4-bit quantization via bitsandbytes (QLoRA)"}
    )
    lora_rank: int = field(
        default=16,
        metadata={"help": "Rank parameter for LoRA adaptation"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Alpha parameter for LoRA adaptation (scaling factor)"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        metadata={"help": "List of module names or regex patterns to apply LoRA to. Use 'all-linear' for all linear layers."}
    )
    # Add bnb quantization config if needed (for 4bit)
    # bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "Quantization type (fp4 or nf4)"})
    # bnb_4bit_compute_dtype: str = field(default="torch.bfloat16", metadata={"help": "Compute dtype for 4-bit base models"})
    # bnb_4bit_use_double_quant: bool = field(default=False, metadata={"help": "Whether to use double quantization"})


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(
        default="SakanaAI/AI-CUDA-Engineer-Archive",
        metadata={"help": "The name of the dataset to use (via the datasets library)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (e.g., 'clean')."}
    )
    pytorch_column: str = field(
        default="PyTorch_Code_Module",
        metadata={"help": "Column name in the dataset containing PyTorch code."}
    )
    cuda_column: str = field(
        default="CUDA_Code",
        metadata={"help": "Column name in the dataset containing target CUDA code."}
    )
    difficulty_column: Optional[str] = field(
        default="Level_ID", # Matches config.json
        metadata={"help": "Optional column name indicating example difficulty (e.g., for stratified sampling)."}
    )
    is_correct_column: Optional[str] = field(
        default="Correct", # Matches config.json
        metadata={"help": "Optional column indicating if the example is a correct translation (for contrastive learning)."}
    )
    max_pytorch_length: int = field(
        default=2048, # Reduced default from original config
        metadata={"help": "Maximum sequence length for PyTorch input code."}
    )
    max_cuda_length: int = field(
        default=3072, # Reduced default from original config
        metadata={"help": "Maximum sequence length for CUDA output code."}
    )
    max_combined_length: Optional[int] = field(
        default=None, # Calculated later if needed
        metadata={"help": "Maximum combined sequence length for model input (pytorch + cuda + formatting). If None, uses max_pytorch_length + max_cuda_length."}
    )
    stratified_sampling: bool = field(
        default=True, # Matches config.json
        metadata={"help": "Enable stratified sampling based on the 'difficulty_column'."}
    )
    use_contrastive: bool = field(
        default=False, # Matches config.json
        metadata={"help": "Enable contrastive learning using 'is_correct_column'."}
    )
    contrastive_loss_weight: float = field(
        default=0.1, # Matches config.json
        metadata={"help": "Weight for the contrastive loss component if use_contrastive is True."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    test_split_size: float = field(
        default=0.1, metadata={"help": "Fraction of the dataset to use for the test split."}
    )
    validation_split_size: float = field(
        default=0.1, metadata={"help": "Fraction of the *test* split to use for validation (e.g., 0.1 means 10% of the test split becomes validation)."}
    )

@dataclass
class DifficultyProgressionArguments:
    """
    Arguments related to difficulty progression callback.
    """
    enable_difficulty_progression: bool = field(
        default=True, # Matches config.json enable=true
        metadata={"help": "Enable the difficulty progression callback (requires stratified_sampling=True)."}
    )
    # Use JSON strings for dictionary arguments to handle complex structures via CLI
    initial_difficulty_weights: str = field(
        default='{"1": 0.6, "2": 0.3, "3": 0.1}', # Using "1", "2", "3" as keys based on likely Level_ID values
        metadata={"help": "Initial sampling weights for difficulty levels as a JSON string (e.g., '{\"1\": 0.6, \"2\": 0.3, \"3\": 0.1}')"}
    )
    final_difficulty_weights: str = field(
        default='{"1": 0.1, "2": 0.3, "3": 0.6}', # Using "1", "2", "3"
        metadata={"help": "Final sampling weights for difficulty levels as a JSON string (e.g., '{\"1\": 0.1, \"2\": 0.3, \"3\": 0.6}')"}
    )
    difficulty_progression_type: str = field(
        default="linear", # Defaulting to linear as it's implemented
        metadata={"help": "Type of weight progression ('linear', 'exponential', 'sigmoid' - currently only 'linear' is implemented)."}
    )
    difficulty_warmup_epochs: float = field(
        default=1.0, # Matches config.json
        metadata={"help": "Number of initial epochs to keep initial weights before starting progression."}
    )
    difficulty_log_every_epoch: int = field(
        default=1, # Matches config.json
        metadata={"help": "Log updated difficulty weights every N epochs."}
    )

    def parse_weights(self, weights_str: str) -> Dict[str, float]:
        """Parses the JSON string weights into a dictionary."""
        try:
            weights = json.loads(weights_str)
            if not isinstance(weights, dict):
                raise ValueError("Weights must be a JSON dictionary.")
            return {str(k): float(v) for k, v in weights.items()} # Ensure keys are strings, values are floats
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format for weights: {weights_str}") from e
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid format within weights dictionary: {weights_str}") from e

    @property
    def initial_weights_dict(self) -> Dict[str, float]:
        return self.parse_weights(self.initial_difficulty_weights)

    @property
    def final_weights_dict(self) -> Dict[str, float]:
        return self.parse_weights(self.final_difficulty_weights)
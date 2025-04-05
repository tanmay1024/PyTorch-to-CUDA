import torch
import logging
import numpy as np
from typing import Dict, Optional, Union
from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)

class DifficultyProgressionCallback(TrainerCallback):
    """
    Callback to dynamically adjust the sampling weights for different difficulty levels
    as training progresses. Starts with higher weights for easy examples and
    gradually shifts towards harder examples.
    """
    def __init__(
        self,
        train_dataset,
        difficulty_levels: Dict[str, int] = {"easy": 0, "medium": 1, "hard": 2},
        initial_weights: Dict[str, float] = {"easy": 0.6, "medium": 0.3, "hard": 0.1},
        final_weights: Dict[str, float] = {"easy": 0.1, "medium": 0.3, "hard": 0.6},
        progression_type: str = "linear",
        warmup_epochs: float = 0.5,
        log_every_epoch: int = 1
    ):
        """
        Initialize the DifficultyProgressionCallback.
        
        Args:
            train_dataset: The training dataset (must be StratifiedSamplingDataset)
            difficulty_levels: Dictionary mapping difficulty names to their numerical level
            initial_weights: Initial sampling weights for each difficulty level
            final_weights: Final sampling weights to reach by the end of training
            progression_type: Type of progression ('linear', 'exponential', or 'sigmoid')
            warmup_epochs: Number of epochs to keep initial weights before starting progression
            log_every_epoch: How often to log the current weights (in epochs)
        """
        self.train_dataset = train_dataset
        self.difficulty_levels = difficulty_levels
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.progression_type = progression_type
        self.warmup_epochs = warmup_epochs
        self.log_every_epoch = log_every_epoch
        
        # Validate inputs
        if not hasattr(self.train_dataset, 'weights'):
            raise ValueError("The train_dataset must be a StratifiedSamplingDataset with a 'weights' attribute")
        
        # Check if all difficulty levels from dataset are covered in the weights
        dataset_difficulties = set(self.train_dataset.difficulty_indices.keys())
        weight_difficulties = set(self.initial_weights.keys())
        
        if not dataset_difficulties.issubset(weight_difficulties):
            missing = dataset_difficulties - weight_difficulties
            raise ValueError(f"Missing weights for difficulty levels: {missing}")
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.total_epochs = None
        self.current_weights = dict(initial_weights)
        
        # Initial log
        logger.info(f"DifficultyProgressionCallback initialized with progression_type={progression_type}")
        logger.info(f"Initial weights: {self.current_weights}")
        logger.info(f"Final weights: {self.final_weights}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.total_epochs = args.num_train_epochs
        logger.info(f"Training started with {self.total_epochs} total epochs")
        logger.info(f"Warmup period: {self.warmup_epochs} epochs with fixed initial weights")
        
        # Set initial weights
        self._update_dataset_weights(self.initial_weights)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        # Round to nearest epoch since state.epoch can be a float
        epoch = round(state.epoch)
        
        # Only update weights if the epoch has changed
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            
            # Calculate and set new weights
            new_weights = self._calculate_weights_for_epoch(state.epoch)
            self._update_dataset_weights(new_weights)
            
            # Log current weights periodically
            if epoch % self.log_every_epoch == 0 or epoch == 1:
                logger.info(f"Epoch {epoch}/{self.total_epochs} - Updated difficulty weights: {new_weights}")
    
    def _calculate_weights_for_epoch(self, epoch: float) -> Dict[str, float]:
        """
        Calculate the sampling weights for the current epoch based on the progression type.
        
        Args:
            epoch: Current training epoch (can be a float)
            
        Returns:
            Dictionary mapping difficulty levels to their new weights
        """
        # If we're in the warmup period, keep initial weights
        if epoch < self.warmup_epochs:
            return dict(self.initial_weights)
        
        # Calculate progress (0 -> 1) from warmup to end of training
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = max(0, min(1, progress))  # Clamp to [0, 1]
        
        # Apply progression function
        factor = progress
        
        # Interpolate between initial and final weights
        new_weights = {}
        for difficulty, initial_weight in self.initial_weights.items():
            final_weight = self.final_weights.get(difficulty, initial_weight)
            new_weights[difficulty] = initial_weight + factor * (final_weight - initial_weight)
        
        # Normalize weights to ensure they sum to 1
        weight_sum = sum(new_weights.values())
        return {k: v / weight_sum for k, v in new_weights.items()}
    
    def _update_dataset_weights(self, new_weights: Dict[str, float]):
        """
        Update the weights in the StratifiedSamplingDataset.
        
        Args:
            new_weights: Dictionary mapping difficulty levels to their new weights
        """
        # Update only weights for difficulties that exist in the dataset
        for difficulty, weight in new_weights.items():
            if difficulty in self.train_dataset.weights:
                self.train_dataset.weights[difficulty] = weight
        
        # Update probabilities in the dataset
        total_weight = sum(self.train_dataset.weights.values())
        self.train_dataset.probs = {
            diff: self.train_dataset.weights.get(diff, 1.0) / total_weight 
            for diff in self.train_dataset.difficulty_indices.keys()
        }
        
        # Store current weights for tracking
        self.current_weights = dict(self.train_dataset.weights)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        # Optional: log evaluation metrics per difficulty level
        pass
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        logger.info(f"Training completed. Final difficulty weights: {self.current_weights}")


# Modified StratifiedSamplingDataset to work with the callback
class StratifiedSamplingDataset(torch.utils.data.Dataset):
    """
    Custom dataset that performs stratified sampling based on difficulty levels.
    Modified to support dynamic weight updates.
    """
    def __init__(self, dataset, difficulty_column, weights):
        self.dataset = dataset
        self.difficulty_column = difficulty_column
        self.weights = weights  # Store as attribute for the callback to modify
        
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
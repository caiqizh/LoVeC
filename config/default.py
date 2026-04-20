from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TrainingConfig:
    # Dataset
    dataset_name: str = "ecqa"
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_seq_length: int = 1024
    lora_rank: int = 64
    load_in_4bit: bool = True
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.5
    target_modules: List[str] = None
    
    # Training args
    use_vllm: bool = True
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    logging_steps: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 8
    max_prompt_length: int = 256
    max_completion_length: int = 32
    num_train_epochs: Optional[int] = None
    max_steps: int = 250
    save_steps: int = 250
    max_grad_norm: float = 0.1
    reward_scaling: float = 5.0
    report_to: str = "none"
    output_dir: str = "outputs"
    wandb_name: str = "default"
    dtype: str = "bfloat16"
    num_train_epochs: int = None
    beta : float = 0.1 # Parameter controlling the deviation from the reference model. Higher β means less deviation from the reference model.

    
    # Reward functions
    reward_funcs: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        
        if self.reward_funcs is None:
            self.reward_funcs = [
                "correctness_reward_func",
                "confidence_reward_func", 
                "int_reward_func",
                "soft_format_reward_func"
            ]
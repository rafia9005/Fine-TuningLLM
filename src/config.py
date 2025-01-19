max_seq_length = 2048
dtype = None
load_in_4bit = True

model_name = "unsloth/llama-3-8b-bnb-4bit"

lora_config = {
    "r": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_alpha": 16,
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
    "random_state": 3407,
    "use_rslora": False,
    "loftq_config": None,
}

training_args = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "max_steps": 1000,
    "learning_rate": 2e-4,
    "fp16": not is_bfloat16_supported(),
    "bf16": is_bfloat16_supported(),
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "output_dir": "output",
}

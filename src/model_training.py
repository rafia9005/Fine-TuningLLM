from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from .config import model_name, max_seq_length, dtype, load_in_4bit, lora_config, training_args
from .data_preprocessing import load_and_preprocess_data

def train_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        **lora_config
    )

    dataset = load_and_preprocess_data()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(**training_args),
    )

    trainer_stats = trainer.train()

    model.save_pretrained("model", tokenizer, quantization_method="f16")

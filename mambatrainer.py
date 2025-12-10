
from mamba.model import MambaTextClassification

from util import preprocess_function, compute_metrics
from mamba.trainer import MambaTrainer
from mamba.trainer import CsvLogCallback

import os
import random
import numpy as np
from huggingface_hub import login
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from transformers import Trainer
from transformers import AutoTokenizer, TrainingArguments

os.environ["TRITON_DISABLE_MMA"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"


os.environ["http_proxy"] = "http://127.0.0.1:8889"
os.environ["https_proxy"] = "http://127.0.0.1:8889"

# os.environ["TRITON_DISABLE_MMA"] = "1"

print("login....")

print("load datasets...")
# imdb = load_dataset("imdb")
# eval_ratio=0.1

# model = MambaTextClassification.from_pretrained("state-spaces/mamba2-130m", num_class=K)
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
# # tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
# tokenizer.pad_token_id = tokenizer.eos_token_id


def mamba_train(dataset, dataset_name, model, tokenizer, epoch):
    # if epoch == 0:
    #     model = MambaTextClassification.from_pretrained("state-spaces/mamba2-130m", num_class=K)
    #     tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

    # datasets = load_from_disk('./text_datasets/cora_after_comm')
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Load the Mamba model from a pretrained model.
    # model = MambaTextClassification.from_pretrained("LeoYML/biomamba-130m")
    # model = MambaTextClassification.from_pretrained("state-spaces/mamba-130m")
    model.to("cuda")

    # Load the tokenizer of the Mamba model from the gpt-neox-20b model.
    # tokenizer = AutoTokenizer.from_pretrained("LeoYML/biomamba-130m")
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    # tokenizer.pad_token_id = tokenizer.eos_token_id


    def tokenize_function(examples):
        tokenized = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
        tokenized['labels'] = examples['label']
        return tokenized


    train_tokenized = train_dataset.map(tokenize_function, batched=True, num_proc=4)  # num_proc 视资源调整
    train_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    test_tokenized = test_dataset.map(tokenize_function, batched=True, num_proc=4)
    test_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # 保存
    tokenized_datasets = DatasetDict({
        'train': train_tokenized,
        'test': test_tokenized
    })
    # tokenized_datasets.save_to_disk("./text_datasets/tokenized_cora")
    # tokenized_datasets = load_from_disk("./text_datasets/tokenized_cora")
    train_data = tokenized_datasets['train']
    test_data = tokenized_datasets['test']

    # Set the pad token id to the eos token id in the tokenizer.
    tokenizer.pad_token_id = tokenizer.eos_token_id

    training_args = TrainingArguments(
        gradient_checkpointing=True,
        fp16=False,
        output_dir=f"papertext_{dataset_name}_after_comm",  # Output folder name
        learning_rate=5e-6,  # cora 5e-5
        per_device_train_batch_size=1,  # Number of training samples per device
        per_device_eval_batch_size=1,  # Number of evaluation samples per device
        num_train_epochs=1,  # Number of training epochs
        warmup_ratio=0.001,  # Ratio of increasing LR during warmup cora 0.001
        lr_scheduler_type="cosine",  # Type of scheduler to decrease LR
        report_to="none",  # "wandb" if you want to log results
        evaluation_strategy="steps",  # Determine the metric for evaluation after each step
        eval_steps=0.1,  # Number of steps between evaluation batches
        save_strategy="steps",  # Determine when to save checkpoints
        save_steps=0.1,  # Number of steps between saving checkpoints
        # logging_dir="./logs",  # 日志目录
        logging_strategy="steps",  # Determine when to log information
        logging_steps=1,  # Number of steps between logging
        push_to_hub=False,  # Push the results to the Hub
        load_best_model_at_end=True,  # Load the model with the best evaluation result during training
    )

    # Initialize the MambaTrainer class to perform the model training process.
    trainer = MambaTrainer(
        model=model,  # Model to train
        train_dataset=train_data,  # Training data
        eval_dataset=test_data,  # Evaluation data
        tokenizer=tokenizer,  # Tokenizer used to encode data
        args=training_args,  # Pre-defined training parameters
        compute_metrics=compute_metrics,  # Function to calculate performance metrics for evaluation
        callbacks=[CsvLogCallback(dataset_name)]
    )

    # Start the training process by calling the train() function on the trainer class.
    trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    # training_args.save(training_args.output_dir)
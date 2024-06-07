import json
import logging
import torch
import argparse
import os
import sys
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cpu')

def load_model_and_tokenizer(model_name):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.to(device) 
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load model and tokenizer. Error: {e}")
        return None, None
    
def tokenize_function(examples, tokenizer):
    inputs = [f"Question: {q}" for q in examples['question']]
    outputs = [a for a in examples['answer']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(outputs, max_length=128, truncation=True, padding="max_length")
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def tokenize_data(tokenizer, dataset):
    return dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

def train_test_split(dataset, test_size):
    return dataset.train_test_split(test_size=test_size)

def set_training_args():
    return TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=2,
    )

def train_model(model, training_args, train_data, val_data):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    trainer.train()
    return trainer

def main():
    script_path = os.path.realpath(sys.argv[0])
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', 
                        type=str, 
                        required=True, 
                        help='Path to the data file')
    args = parser.parse_args()
    data_path = args.data_path

    tokenizer, model = load_model_and_tokenizer("gpt2-medium")
    if tokenizer is None or model is None:
        logger.error("Failed to load. Exiting.")
        return

    data_file_path = data_path
    with open(data_file_path, 'r') as f:
        data = json.load(f)
    cleaned_data = [row for row in data if row['question'] is not None and row['answer'] is not None]
    questions = [row['question'] for row in cleaned_data]
    answers = [row['answer'] for row in cleaned_data]
    cleaned_data_dict = {'question': questions, 'answer': answers}
    cleaned_dataset = Dataset.from_dict(cleaned_data_dict)

    tokenized_dataset = tokenize_data(tokenizer, cleaned_dataset)
    if tokenized_dataset is None:
        logger.error("Failed to tokenize data. Exiting.")
        return

    train_data = train_test_split(tokenized_dataset, test_size=0.2)

    training_args = set_training_args()
    if training_args is None:
        logger.error("Failed to set training parameters. Exiting.")
        return

    trained_model = train_model(model, training_args, train_data['train'], train_data['test'])
    if trained_model is not None:
        logger.info("Fine-tuning completed successfully.")
        results = trained_model.evaluate()
        logger.info(f"Evaluation results: {results}")

        model_save_path = f'{script_path}/model'   
        tokenizer_save_path = f'{script_path}/tokenizer'
    
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(tokenizer_save_path)

        logger.info(f"Model saved to {model_save_path}")
        logger.info(f"Tokenizer saved to {tokenizer_save_path}")
    else:
        logger.error("Fine-tuning failed.")

if __name__ == "__main__":
    main()
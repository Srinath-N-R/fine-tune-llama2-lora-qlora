import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import yaml
import argparse

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(config):
    # Unpack configurations
    model_name = config['model']['name']
    dataset_name = config['dataset']['name']
    new_model = config['model']['new_name']
    
    lora_r = config['lora']['r']
    lora_alpha = config['lora']['alpha']
    lora_dropout = config['lora']['dropout']
    
    output_dir = config['training']['output_dir']
    num_train_epochs = config['training']['num_train_epochs']
    per_device_train_batch_size = config['training']['train_batch_size']
    per_device_eval_batch_size = config['training']['eval_batch_size']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = config['training']['weight_decay']
    max_grad_norm = config['training']['max_grad_norm']
    optim = config['training']['optimizer']
    lr_scheduler_type = config['training']['lr_scheduler_type']
    seed = config['training']['seed']
    max_steps = config['training']['max_steps']
    warmup_ratio = config['training']['warmup_ratio']
    group_by_length = config['training']['group_by_length']
    save_steps = config['training']['save_steps']
    logging_steps = config['training']['logging_steps']
    
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")
    
    # Check GPU compatibility with bfloat16
    if torch.backends.mps.is_available():
        print("="*80)
        print("Your system supports MPS for acceleration!")
        print("="*80)
    else:
        print("MPS is not available. Using CPU.")
        
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load QLoRA config
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Training arguments
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        max_steps=max_steps,
        report_to="wandb",
        seed=seed,
    )
    
    # Initialize Trainer
    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        peft_config = peft_config,
        args = training_arguments,
        tokenizer = tokenizer,
    )
    
    # Start training
    trainer.train()
    
    # Save the trained model
    trainer.model.save_pretrained(new_model)
    
    # Run text generation pipeline
    logging.set_verbosity(logging.CRITICAL)
    prompt = config['testing']['prompt']
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=config['testing']['max_length'])
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])
    
    # Merge LoRA weights with base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=config['training']['device_map'],
    )
    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()
    
    # Save the merged model
    model.save_pretrained(config['training']['final_model_path'])
    
    # Reload tokenizer to save it
    tokenizer.save_pretrained(config['training']['final_model_path'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tune Llama 2 with LoRA and QLoRA")
    parser.add_argument('--config', type=str, default='training_config.yaml', help='Path to the config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    main(config)

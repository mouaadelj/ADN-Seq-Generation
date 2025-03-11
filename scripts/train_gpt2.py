from transformers import GPT2Config, GPT2TokenizerFast, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from tokenizers import ByteLevelBPETokenizer, pre_tokenizers
from datasets import load_dataset
from pathlib import Path
import os


def train_gpt2(training_data_pth, test_data_pth, model_dir = '../models/GPT2'):
    
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    
    model_path = os.path.join(model_dir, "pytorch_model.bin")
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    if os.path.exists(model_path):

        print(f"Model found at {model_dir}, loading it...")
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
    else:
    
        configuration = GPT2Config()
        #Path(model_dir).mkdir(parents=True, exist_ok=True)
        configuration.save_pretrained(model_dir)

        flex_dataset = load_dataset("csv", data_files={'train': training_data_pth, 'test':test_data_pth})
        tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        def concat_seq(Data):
            Data["seq"] = Data["starting_seq"] + Data["binding"]
            return Data

        def batch_iterator(batch_size=256):
            for i in range(0,len(flex_dataset['train']['starting_seq']), batch_size):
                yield list(flex_dataset['train']['starting_seq'][i: i+batch_size])

        flex_dataset1 = flex_dataset.map(concat_seq, batched=True, num_proc=4, remove_columns=['seq'])


        tokenizer= ByteLevelBPETokenizer()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        tokenizer.train_from_iterator(batch_iterator(), vocab_size= configuration.vocab_size, min_frequency=2, show_progress=True,  special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"])
        
        tokenizer_dir='../models/GPT2/tokenizer'
        tokenizer.save_model(f"{tokenizer_dir}")
        

        tokenizer=GPT2TokenizerFast.from_pretrained('../models/GPT2/tokenizer', max_len=16)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        def tokenize_function(examples):
            return final_tokenizer(examples['seq'], truncation=True, max_length=16)    
        
        tokenized_dataset = flex_dataset1.map(tokenize_function, batched=True, num_proc=4, remove_columns=['seq'])

        model = AutoModelForCausalLM.from_config(configuration)
        Data_collator = DataCollatorForLanguageModeling(tokenizer =final_tokenizer, mlm=False)


        training_args = TrainingArguments(
            output_dir=model_dir,
            overwrite_output_dir=True,
            num_train_epochs=6,
            per_device_train_batch_size=256,
            save_total_limit=2,
            logging_steps=500,
            learning_rate=2e-5,
            evaluation_strategy="epoch",
            weight_decay=0.01,
        )

        trainer = Trainer(
            model = model,
            args=training_args,
            train_dataset = tokenized_dataset['train'],
            eval_dataset = tokenized_dataset['test'],
            data_collator = Data_collator
        )

        trainer.train()
        trainer.save_model(model_dir)

    return model, tokenizer

if __name__ == "__main__":
    train_gpt2("data/training_data.csv", "../models/GPT2")
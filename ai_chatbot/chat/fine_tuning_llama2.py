import os
import torch
import pandas as pd 
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import Dataset
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from transformers import TrainerCallback
import gc
from ai_chatbot.language_model import LLM_PATH

Drive_TRAINER_PATH="static/fine_tuned_model/"


class FineTuningLlama2():
    
    def __init__(self, new_model_name):
        self.__lora_r = 64
        self.__lora_alpha = 16 
        self.__lora_dropout = 0.1
        self.__use_4bit = True
        self.__bnb_4bit_compute_dtype = "float16"
        self.__bnb_4bit_quant_type = "nf4"
        self.__use_nested_quant = False
        self.__output_dir = "static/results"
        self.__num_train_epochs = 1
        self.__fp16 = False
        self.__bf16 = True
        self.__per_device_train_batch_size = 2
        self.__per_device_eval_batch_size = 2
        self.__gradient_accumulation_steps = 1
        self.__gradient_checkpointing = True
        self.__max_grad_norm = 0.3
        self.__learning_rate = 2e-4
        self.__weight_decay = 0.001
        self.__optim = "paged_adamw_32bit"
        self.__lr_scheduler_type = "cosine"
        self.__max_steps = -1
        self.__warmup_ratio = 0.03
        self.__group_by_length = True
        self.__save_steps=25
        self.__logging_steps=25
        self.__max_seq_length = 1024
        self.__packing = False
        self.__llm_model =  LLM_PATH
        self.__new_model_name = new_model_name
        self.__device_map = "cuda:0"


    #Create lama2 format Ds
    def create_prompt(self,sample):
        bos_token = "<s>"
        system_message = "You are  a helpful and truthful assistant. Your answers should be truthful,honest and unbiased. Answer it as succinctly as possible."
        response = str(sample["response"])
        input_ = str(sample["input"])
        eos_token = "</s>###>"

        full_prompt = ""
        full_prompt += bos_token
        full_prompt += "### Instruction:"
        full_prompt += "\n" + system_message
        full_prompt += "\n\n### Input:"
        full_prompt += "\n" + input_
        full_prompt += "\n\n### Response:"
        full_prompt += "\n" + response
        full_prompt += eos_token

        return full_prompt

    def transform_input_file(self,file_url):
        input_df = pd.read_csv(file_url, encoding="utf-8")
        if not 'input' in input_df.columns and not 'response' in input_df.columns:
            return pd.DataFrame()

        input_df['text'] = input_df.apply(lambda row: self.create_prompt(row), axis=1)
        return input_df
    
    def load_tokenizer_and_model(self,input_df):
        dataset = Dataset.from_pandas(input_df)
        compute_dtype = getattr(torch, self.__bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.__use_4bit,
            bnb_4bit_quant_type=self.__bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.__use_nested_quant,
        )
        ##del previous model 
        ##load the model from saved model path 
        ##del previous model 
        ##load the model from saved model path 
        self.__model = AutoModelForCausalLM.from_pretrained(self.__llm_model,
            quantization_config=bnb_config,
            device_map=self.__device_map
        )
        self.__model.config.use_cache = False
        self.__model.config.pretraining_tp = 1
        # Load LLaMA tokenizer
        self.__tokenizer = AutoTokenizer.from_pretrained(self.__llm_model)
        self.__tokenizer.pad_token = self.__tokenizer.eos_token
        self.__tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
        self.__tokenizer.pad_token = self.__tokenizer.eos_token
                
        # Load LoRA configuration
        peft_config = LoraConfig(
            lora_alpha=self.__lora_alpha,
            lora_dropout=self.__lora_dropout,
            r=self.__lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Set training parameters
        training_arguments = TrainingArguments(
            output_dir=self.__output_dir,
            num_train_epochs=self.__num_train_epochs,
            per_device_train_batch_size=self.__per_device_train_batch_size,
            gradient_accumulation_steps=self.__gradient_accumulation_steps,
            optim=self.__optim,
            save_steps=self.__save_steps,
            logging_steps=self.__logging_steps,
            learning_rate=self.__learning_rate,
            weight_decay=self.__weight_decay,
            fp16=self.__fp16,
            bf16=self.__bf16,
            max_grad_norm=self.__max_grad_norm,
            max_steps=self.__max_steps,
            warmup_ratio=self.__warmup_ratio,
            group_by_length=self.__group_by_length,
            lr_scheduler_type=self.__lr_scheduler_type,
        )

        # Set supervised fine-tuning parameters
        trainer = SFTTrainer(
            model=self.__model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=self.__max_seq_length,
            tokenizer=self.__tokenizer,
            args=training_arguments,
            packing=self.__packing
        )

        return trainer
    def run_trainer(self,trainer):
        torch.cuda.empty_cache()
        output = trainer.train()
        return trainer, output

    def save_trainer_model(self,trainer):
        trainer.model.save_pretrained(Drive_TRAINER_PATH)
        trainer.tokenizer.save_pretrained(Drive_TRAINER_PATH)
        del trainer.model
        del trainer.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        

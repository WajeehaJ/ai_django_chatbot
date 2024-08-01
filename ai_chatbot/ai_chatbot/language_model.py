
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
from langchain_huggingface import HuggingFacePipeline
from torch import cuda, LongTensor, FloatTensor
from langchain_community.embeddings import HuggingFaceEmbeddings
#Stopping criteria custom function
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
import gc 
import os
FINE_TUNED_LLM_PATH=r"/home/najeed/NED_Chatbot/ai_chatbot/static/fine_tuned_model"
LLM_PATH = r"/home/najeed/finetuningdata/Fine_tune_llama2_model_with_NED_dataset/llama_ned_fine_tune"

class LanguageModel:
    _instance = None
    @staticmethod
    def get_instance():
        if LanguageModel._instance is None:
            print("Initializing the model")
            LanguageModel()
        else:
            print("Returning old instance ")
        return LanguageModel._instance
    
    @staticmethod
    def is_instance_exits():
        if LanguageModel._instance is None:
            return False
        else:
            return True
    def __init__(self):
        if LanguageModel._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.model_pipe, self.model, self.tokenizer = self.load_model()
            self.embeddings = self.load_embedding()
            LanguageModel._instance = self
    
    def create_stopping_criteria(self,stop_words, tokenizer, device):
        class StoppingCriteriaSub(StoppingCriteria):
            def __init__(self, stops = [], device=device, encounters = 1):
                super().__init__()
                self.stops = stops = [stop.to(device) for stop in stops]

            def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> bool:
                last_token = input_ids[0][-1]
                for stop in self.stops:
                    if tokenizer.decode(stop) == tokenizer.decode(last_token):
                        return True
                return False

        stop_word_ids = [tokenizer(stop_word,
                                return_tensors="pt", 
                                add_special_tokens=False)["input_ids"].squeeze() 
                                for stop_word in stop_words]

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_word_ids)])
        return stopping_criteria

    def load_model(self):
        dir = os.listdir(FINE_TUNED_LLM_PATH) 
        MODEL_PATH = LLM_PATH if len(dir) == 0 else FINE_TUNED_LLM_PATH  
        # Load the locally downloaded model here
        torch.cuda.empty_cache() 
        model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token   
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

        stop_words_list = ["[/INST]</s>", "</s>","[/INST]", "</s>",  "</s>###>", "</s>###> ", "<###>"]
        stopping_criteria = None    
        if stop_words_list is not None:
            stopping_criteria = self.create_stopping_criteria(stop_words_list, tokenizer, device)
        text_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature = 0.1,
        repetition_penalty = 1.2,
        return_full_text=False,
        max_new_tokens=250,
        stopping_criteria = stopping_criteria,
        do_sample=True)

        llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})
        return llm, model, tokenizer
    

    def load_embedding(self):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                        model_kwargs={'device': 'cuda'}, cache_folder=r"/home/najeed/NED_Chatbot/ai_chatbot/embeddings_path")
        return embeddings
    
        
    def delete_model(self):
        print("delete LLM invokded!")
        if not self._instance:
            del self.model
            del self.model_pipe
            del self.tokenizer
            del self.embeddings


        gc.collect()
        torch.cuda.empty_cache() 
        gc.collect()
            
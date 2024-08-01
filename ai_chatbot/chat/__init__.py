import os  
import signal  
import sys  
from ai_chatbot.language_model import * 
from chat.llm_vector_db import *

llmVectorDB().create_vector_db()


def signal_handler(sig, frame):
    if LanguageModel.is_instance_exits():
        llm_model = LanguageModel.get_instance()
        print(f'model: {llm_model}')
        llm_model.delete_model()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


 


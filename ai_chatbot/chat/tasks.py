
import time
from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer
from ai_chatbot.language_model import *
from .chat_qabot import *
from chat.llm_vector_db import *
from chat.fine_tuning_llama2 import *
channel_layer = get_channel_layer()


@shared_task
def get_response(channel_name, input_data, user_history):
    print("Getting response")
    llm_model = LanguageModel.get_instance()
    bot = ChatQAbot(llm_model)
    qa_result = bot.get_bot_reply(input_data["text"], user_history)
    print(qa_result)
    async_to_sync(channel_layer.send)(
        channel_name,
        {
            "type": "chat.message",
            "text": {"msg": qa_result, "source": "bot"},
        },
    )

@shared_task
def add_documents_to_VectorDB():
    vector_db  = llmVectorDB()
    result = vector_db.add_documents_to_vector_DB()
    print("Documents add to vector db")
    return result


@shared_task
def run_fine_tuning_LLM(input_file_path, new_model_name):
    #Unload the existing LLM model before fine tuning the model
    
    if LanguageModel.is_instance_exits():
        llama2_model = LanguageModel.get_instance()
        llama2_model.delete_model()
        time.sleep(5)
    
    fine_tune_llm = FineTuningLlama2(new_model_name)
    fine_tune_DS = fine_tune_llm.transform_input_file(input_file_path)
    if fine_tune_DS.empty:
        return False
    trainer = fine_tune_llm.load_tokenizer_and_model(fine_tune_DS)
    trainer, output = fine_tune_llm.run_trainer(trainer)
    print(output)
    result = fine_tune_llm.save_trainer_model(trainer)
    time.sleep(5)
    return True





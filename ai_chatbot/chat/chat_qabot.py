from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
import torch
import re
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from torch import cuda, LongTensor, FloatTensor
import gc

DB_FAISS_PATH = 'vectorstore/db_faiss'
custom_prompt_template = """<s><<SYS>>You are a helpful, respectful and honest assistant. If the context is given then make up the answer  the context provided with the query or you can answer the query using history of previous conversations, history is delimited by <hs></hs>.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.<</SYS>>
    [INST]
    <hs>{chat_history}</hs>
    ### context:
    {context}
    ### Input:
    {question}
    [/INST]</s>
    """

class ChatQAbot():
    
    def __init__(self, llm_model):
        print("Loading LLM")
        self.ned_llm = llm_model.model_pipe
        self.hugging_embdeddings = llm_model.embeddings
        
    #Retrieval QA Chain
    def retrieval_qa_chain(self, prompt, db):

        qa_chain = ConversationalRetrievalChain.from_llm(llm=self.ned_llm, retriever = db.as_retriever(search_kwargs={'k': 2}),
                                        return_source_documents=False, verbose=True, chain_type="stuff", get_chat_history=lambda h : h,
                                        combine_docs_chain_kwargs={'prompt': prompt}
                                        
                                        )                                  
                                        
        return qa_chain

    def set_custom_prompt(self):
        """
        Prompt template for QA retrieval for each vectorstore
        """
        prompt = PromptTemplate(template=custom_prompt_template,
                                input_variables=['context', 'question', 'chat_history'])
        return prompt
    
   
    #QA Model Function
    def qa_bot(self):
        db = FAISS.load_local(DB_FAISS_PATH, self.hugging_embdeddings , allow_dangerous_deserialization=True)
        qa_prompt = self.set_custom_prompt()
        qa = self.retrieval_qa_chain( qa_prompt, db)

        return qa


    #output function
    def get_bot_reply(self,query, chat_history):
        qa_result = self.qa_bot()
        response = qa_result.invoke({'question': query,'chat_history':chat_history })
        return  response["answer"]
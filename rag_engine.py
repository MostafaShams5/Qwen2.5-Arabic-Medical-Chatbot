"""
Description: Inference Engine using LlamaCpp and ChromaDB
Author: "Mostafa Shams"
"""

import os
import sys
import logging
from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_FILENAME = "qwen2.5-3b-instruct.Q5_K_M.gguf" 

CONFIDENCE_THRESHOLD = 0.72    
RETRIEVAL_K = 2               
MAX_CONTEXT_WINDOW = 2048 
MAX_OUTPUT_TOKENS = 512

SYSTEM_PROMPT = """You only answer MEDICAL questions and Answer in Arabic ONLY. 

MUST FOLLOW INSTRUCTIONS:
1. If the User Question is not strictly related to the medical context provided, or if the context is empty, YOU MUST REFUSE with the Refusal message: "عذراً، هذا السؤال خارج نطاق تخصصي الطبي."
2. Always answer in Arabic.
"""

CHAT_TEMPLATE = f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
Context (chunks ~400 tokens):
{{context}}

Question:
{{question}}
<|im_end|>
<|im_start|>assistant
"""

class FastRAGEngine:
    def __init__(self):
        self.qa_chain = None
        self._initialize()

    def _initialize(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, MODEL_FILENAME)
        db_path = os.path.join(base_dir, "medical_rag_db")

        embedding_fn = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_fn)
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.5,        
            top_p=0.9,           
            max_tokens=MAX_OUTPUT_TOKENS, 
            n_ctx=MAX_CONTEXT_WINDOW,
            n_batch=1024,     
            n_gpu_layers=0,          
            n_threads=os.cpu_count(),
            callback_manager=callback_manager,
            verbose=False,
            streaming=True
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": RETRIEVAL_K, 
                "score_threshold": CONFIDENCE_THRESHOLD
            }
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=CHAT_TEMPLATE,
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def query(self, text):
        try:
            print("\n", end="") 
            response = self.qa_chain.invoke({"query": text})
            
            if not response['source_documents']:
                print("No relevant medical context found.")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    app = FastRAGEngine()
    while True:
        q = input("\nQuery: ")
        if q.lower() in ['q', 'exit']: break
        app.query(q)

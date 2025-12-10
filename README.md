# Qwen2.5 Arabic Medical Chatbot

**NOTE: This project is not complete. I am still working on improving the model and the data.**

## Overview
This is my graduation project. The goal is to create a medical chatbot that speaks Arabic. I used the Qwen 2.5 model (3 Billion parameters) as the base. I applied RAG (Retrieval Augmented Generation) so the bot can search a medical database before answering.

I trained and tested this project using a Tesla T4 GPU.

**Model Download:**
You can find the trained GGUF model here: [Shams03/Qwen2.5-3B-Medical-Arabic-GGUF](https://huggingface.co/Shams03/Qwen2.5-3B-Medical-Arabic-GGUF)

## Performance Results

Here are the numbers from my training and testing process:

| Metric | Value |
| :--- | :--- |
| **Training Dataset Size** | 2004 examples |
| **Training Epochs** | 3 |
| **Baseline BLEU Score** (Before Training) | 0.029 |
| **Final BLEU Score** (After Training) | 0.3296 |
| **Original Model Memory (VRAM)** | 5.76 GB |
| **Quantized Model Memory (VRAM)** | 2.57 GB |

## Project Files

1.  **vector_db_builder.ipynb**
    This notebook organizes the data for the search engine.
    *   It reads text from `medical_rag.json`.
    *   It cuts the text into chunks of 1000 characters with an overlap of 130 characters.
    *   It uses the `intfloat/multilingual-e5-large` model to turn text into numbers (embeddings).
    *   It saves everything into a ChromaDB vector database.

2.  **llm_finetuner.ipynb**
    This notebook trains the AI to understand medical questions better.
    *   I used the Unsloth library to speed up training on the T4 GPU.
    *   I used LoRA configuration with a rank of 16 and alpha of 16.
    *   After training, I converted the model to GGUF format (q5_k_m) to make it smaller and faster.

3.  **rag_engine.py**
    This is the python script that runs the chatbot.
    *   It loads the GGUF model and the database.
    *   When a user asks a question, it finds the 2 most relevant pieces of text.
    *   It filters out results with a confidence score lower than 0.72.
    *   The system instructions force the model to answer only in Arabic and refuse non-medical questions.

## How to Run
1.  Run `vector_db_builder.ipynb` to generate the database files.
2.  Run `llm_finetuner.ipynb` to train the model and get the `.gguf` file (or download it from the link above).
3.  Run `python rag_engine.py` in your terminal to start chatting.

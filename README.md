# hfdl
Huggingface Downloader

Forked from https://github.com/oobabooga/text-generation-webui and stripped of everything but the logic to download files. Then slightly refactored to run with uvicorn instead of Gradio.

How to run:

` pip install -r requirements.txt`  
`uvicorn main:app --reload --host 0.0.0.0 --port 7869` 
Adjust Port and IP as needed.

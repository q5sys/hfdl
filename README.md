# hfdl
Huggingface Downloader

Forked from https://github.com/oobabooga/text-generation-webui and stripped of everything but the logic to download files. Then slightly refactored to run with uvicorn instead of Gradio.

There are times when I'm browsing that I find a neat model that I want to try and I dont want he hassle of popping a shell into my server to pull down a repo via the CLI.
Oobabooga has a nice little model downloader in it, but it's a bit overkill to run all of oobabooga on my machine constantly just for the random times when I want to download something off HF.
So I ripped the code out for their downloader and wrapped it in a braindead simple Uvicorn UI. This is about as basic as it gets. It operates the same way as the OB downloader, except you have to define the path (so you can save it wherever for testing). It takes a page out of the Unix-Philosophy; it does one thing, does it well, and it does nothing else.

How to run:

` pip install -r requirements.txt`  
`uvicorn main:app --reload --host 0.0.0.0 --port 7869` 
Adjust Port and IP as needed.

Yes it really is as simple as it seems:
![image](https://github.com/q5sys/hfdl/assets/4654247/924ba15c-eb1e-4493-aafa-bb828cf75453)

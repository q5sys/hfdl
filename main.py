from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.encoders import jsonable_encoder
import importlib
import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path
import traceback

import requests
import tqdm
from requests.adapters import HTTPAdapter
from tqdm.contrib.concurrent import thread_map

base = "https://huggingface.co"
templates = Jinja2Templates(directory="templates")


class ModelDownloader:
    def __init__(self, max_retries=5):
        self.session = requests.Session()
        if max_retries:
            self.session.mount('https://cdn-lfs.huggingface.co', HTTPAdapter(max_retries=max_retries))
            self.session.mount('https://huggingface.co', HTTPAdapter(max_retries=max_retries))

        if os.getenv('HF_USER') is not None and os.getenv('HF_PASS') is not None:
            self.session.auth = (os.getenv('HF_USER'), os.getenv('HF_PASS'))

        try:
            from huggingface_hub import get_token

            token = get_token()
        except ImportError:
            token = os.getenv("HF_TOKEN")

        if token is not None:
            self.session.headers = {'authorization': f'Bearer {token}'}

    def sanitize_model_and_branch_names(self, model, branch):
        print("Sanitizing model and branch names") 
        if model[-1] == '/':
            model = model[:-1]

        if model.startswith(base + '/'):
            model = model[len(base) + 1:]

        model_parts = model.split(":")
        model = model_parts[0] if len(model_parts) > 0 else model
        branch = model_parts[1] if len(model_parts) > 1 else branch

        if branch is None:
            branch = "main"
        else:
            pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
            if not pattern.match(branch):
                raise ValueError(
                    "Invalid branch name. Only alphanumeric characters, period, underscore and dash are allowed.")

        return model, branch


    def get_download_links_from_huggingface(self, model, branch, text_only=False, specific_file=None):
        print("Getting the download links from Hugging Face")
        page = f"/api/models/{model}/tree/{branch}"
        cursor = b""

        links = []
        sha256 = []
        classifications = []
        has_pytorch = False
        has_pt = False
        has_gguf = False
        has_safetensors = False
        is_lora = False
        while True:
            url = f"{base}{page}" + (f"?cursor={cursor.decode()}" if cursor else "")
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            content = r.content

            dict = json.loads(content)
            if len(dict) == 0:
                break

            for i in range(len(dict)):
                fname = dict[i]['path']
                if specific_file not in [None, ''] and fname != specific_file:
                    continue

                if not is_lora and fname.endswith(('adapter_config.json', 'adapter_model.bin')):
                    is_lora = True

                is_pytorch = re.match(r"(pytorch|adapter|gptq)_model.*\.bin", fname)
                is_safetensors = re.match(r".*\.safetensors", fname)
                is_pt = re.match(r".*\.pt", fname)
                is_gguf = re.match(r'.*\.gguf', fname)
                is_tiktoken = re.match(r".*\.tiktoken", fname)
                is_tokenizer = re.match(r"(tokenizer|ice|spiece).*\.model", fname) or is_tiktoken
                is_text = re.match(r".*\.(txt|json|py|md)", fname) or is_tokenizer
                if any((is_pytorch, is_safetensors, is_pt, is_gguf, is_tokenizer, is_text)):
                    if 'lfs' in dict[i]:
                        sha256.append([fname, dict[i]['lfs']['oid']])

                    if is_text:
                        links.append(f"https://huggingface.co/{model}/resolve/{branch}/{fname}")
                        classifications.append('text')
                        continue

                    if not text_only:
                        links.append(f"https://huggingface.co/{model}/resolve/{branch}/{fname}")
                        if is_safetensors:
                            has_safetensors = True
                            classifications.append('safetensors')
                        elif is_pytorch:
                            has_pytorch = True
                            classifications.append('pytorch')
                        elif is_pt:
                            has_pt = True
                            classifications.append('pt')
                        elif is_gguf:
                            has_gguf = True
                            classifications.append('gguf')

            cursor = base64.b64encode(f'{{"file_name":"{dict[-1]["path"]}"}}'.encode()) + b':50'
            cursor = base64.b64encode(cursor)
            cursor = cursor.replace(b'=', b'%3D')

        # If both pytorch and safetensors are available, download safetensors only
        if (has_pytorch or has_pt) and has_safetensors:
            for i in range(len(classifications) - 1, -1, -1):
                if classifications[i] in ['pytorch', 'pt']:
                    links.pop(i)

        # For GGUF, try to download only the Q4_K_M if no specific file is specified.
        # If not present, exclude all GGUFs, as that's likely a repository with both
        # GGUF and fp16 files.
        if has_gguf and specific_file is None:
            has_q4km = False
            for i in range(len(classifications) - 1, -1, -1):
                if 'q4_k_m' in links[i].lower():
                    has_q4km = True

            if has_q4km:
                for i in range(len(classifications) - 1, -1, -1):
                    if 'q4_k_m' not in links[i].lower():
                        links.pop(i)
            else:
                for i in range(len(classifications) - 1, -1, -1):
                    if links[i].lower().endswith('.gguf'):
                        links.pop(i)

        is_llamacpp = has_gguf and specific_file is not None
        return links, sha256, is_lora, is_llamacpp

    def get_output_folder(self, model, branch, is_lora, is_llamacpp=False, base_folder=None):
        print("Getting the output folder")
        if base_folder is None:
            base_folder = 'models' if not is_lora else 'loras'

        # If the model is of type GGUF, save directly in the base_folder
        if is_llamacpp:
            return Path(base_folder)

        output_folder = f"{'_'.join(model.split('/')[-2:])}"
        if branch != 'main':
            output_folder += f'_{branch}'

        output_folder = Path(base_folder) / output_folder
        return output_folder

    def get_single_file(self, url, output_folder, start_from_scratch=False):
        print(f"Downloading {url} to {output_folder}")
        filename = Path(url.rsplit('/', 1)[1])
        output_path = output_folder / filename
        headers = {}
        mode = 'wb'
        if output_path.exists() and not start_from_scratch:

            # Check if the file has already been downloaded completely
            r = self.session.get(url, stream=True, timeout=10)
            total_size = int(r.headers.get('content-length', 0))
            if output_path.stat().st_size >= total_size:
                return

            # Otherwise, resume the download from where it left off
            headers = {'Range': f'bytes={output_path.stat().st_size}-'}
            mode = 'ab'

        with self.session.get(url, stream=True, headers=headers, timeout=10) as r:
            r.raise_for_status()  # Do not continue the download if the request was unsuccessful
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB

            tqdm_kwargs = {
                'total': total_size,
                'unit': 'iB',
                'unit_scale': True,
                'bar_format': '{l_bar}{bar}| {n_fmt:6}/{total_fmt:6} {rate_fmt:6}'
            }

            if 'COLAB_GPU' in os.environ:
                tqdm_kwargs.update({
                    'position': 0,
                    'leave': True
                })

            with open(output_path, mode) as f:
                with tqdm.tqdm(**tqdm_kwargs) as t:
                    count = 0
                    for data in r.iter_content(block_size):
                        t.update(len(data))
                        f.write(data)
                        if total_size != 0 and self.progress_bar is not None:
                            count += len(data)
                            self.progress_bar(float(count) / float(total_size), f"{filename}")

    def start_download_threads(self, file_list, output_folder, start_from_scratch=False, threads=4):
        thread_map(lambda url: self.get_single_file(url, output_folder, start_from_scratch=start_from_scratch), file_list, max_workers=threads, disable=True)

    def download_model_files(self, model, branch, links, sha256, output_folder, progress_bar=None, start_from_scratch=False, threads=4, specific_file=None, is_llamacpp=False):
        self.progress_bar = progress_bar
        if self.progress_bar is not None and callable(self.progress_bar):
            print("Progress bar is callable")

        # Create the folder and writing the metadata
        output_folder = Path(output_folder)  # Convert output_folder to a Path object
        try:
            output_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error while creating directory: {e}")
        #print(f"Type of output_folder: {type(output_folder)}")  # Added this line to debug the type of output_folder

        if not is_llamacpp:
            metadata = f'url: https://huggingface.co/{model}\n' \
                       f'branch: {branch}\n' \
                       f'download date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'

            sha256_str = '\n'.join([f'    {item[1]} {item[0]}' for item in sha256])
            if sha256_str:
                metadata += f'sha256sum:\n{sha256_str}'

            metadata += '\n'
            (output_folder / 'huggingface-metadata.txt').write_text(metadata)

        if specific_file:
            print(f"Downloading {specific_file} to {output_folder}")
        else:
            print(f"Downloading the model to {output_folder}")

        self.start_download_threads(links, output_folder, start_from_scratch=start_from_scratch, threads=threads)


    def check_model_files(self, model, branch, links, sha256, output_folder):
        # Validate the checksums
        validated = True
        for i in range(len(sha256)):
            fpath = (output_folder / sha256[i][0])

            if not fpath.exists():
                print(f"The following file is missing: {fpath}")
                validated = False
                continue

            with open(output_folder / sha256[i][0], "rb") as f:
                file_hash = hashlib.file_digest(f, "sha256").hexdigest()
                if file_hash != sha256[i][1]:
                    print(f'Checksum failed: {sha256[i][0]}  {sha256[i][1]}')
                    validated = False
                else:
                    print(f'Checksum validated: {sha256[i][0]}  {sha256[i][1]}')

        if validated:
            print('[+] Validated checksums of all model files!')
        else:
            print('[-] Invalid checksums. Rerun download-model.py with the --clean flag.')
            

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    # Serve the HTML form to the user  # uvicorn main:app --reload --host 0.0.0.0 --port 7869
    return templates.TemplateResponse("form.html", {"request": request})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content=jsonable_encoder({"detail": "Missing required fields", "body": exc.errors()}),
    )


# ...

@app.post("/download-model/")
async def download_model(request: Request,
                         path: str = Form(...),
                         repo: str = Form(...),
                         specific_file: str = Form(None)):
    form_data = {"path": path, "repo": repo, "specific_file": specific_file}
    print(f"Form data: {form_data}")
    downloader = ModelDownloader(max_retries=5)
    
    if not path:
        # Read the "path" variable from settings.txt if path is empty
        settings_file = os.path.join(os.path.dirname(__file__), "settings.txt")
        with open(settings_file, "r") as f:
            path = f.read().strip()
    
    os.makedirs(path, exist_ok=True)
    print(f"Download path: {path}")

    # Sanitize and prepare model and branch names
    model, branch = downloader.sanitize_model_and_branch_names(repo, None)
    print(f"Model: {model}, Branch: {branch}")
    # Get download links (modify according to your needs and method definitions)

    try:
        # Determine output folder based on form input and download the model
        # This assumes you have a method to determine the output folder and download the model
        output_folder = os.path.join(path, model)  # Example, adjust as needed
        print(f"Output folder: {output_folder}")
        #download_model_wrapper(path, repo, specific_file)
        links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=False, specific_file=specific_file)
        downloader.download_model_files(model, branch, links, sha256, output_folder, threads=4, is_llamacpp=is_llamacpp)
        
        message = "Model downloaded successfully."
    except Exception as e:
        message = f"Failed to download model: {str(e)}"

    # Render response with a success or error message
    return templates.TemplateResponse("form.html", {"request": request, "message": message})

#def create_event_handlers():
#    shared.gradio['download_model_button'].click(download_model_wrapper, gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)
#    shared.gradio['get_file_list'].click(partial(download_model_wrapper, return_links=True), gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)


def download_model_wrapper(path, repo_id, specific_file):
    try:
        def progress(value):
            print(f"Progress: {value * 100}%")

        progress(0.0)
        downloader = importlib.import_module("download-model").ModelDownloader()
        model, branch = downloader.sanitize_model_and_branch_names(repo_id, None)
        yield ("Getting the download links from Hugging Face")
        links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=False, specific_file=specific_file)
        # if return_links:
        #    yield '\n\n'.join([f"`{Path(link).name}`" for link in links])
        #    return
        check = True  # Define the variable "check" before using it


        yield ("Getting the output folder")
        base_folder = path
        output_folder = downloader.get_output_folder(model, branch, is_lora, is_llamacpp=is_llamacpp, base_folder=base_folder)
        if check:
            progress(0.5)
            yield ("Checking previously downloaded files")
            downloader.check_model_files(model, branch, links, sha256, output_folder)
            progress(1.0)
        else:
            yield (f"Downloading file{'s' if len(links) > 1 else ''} to `{output_folder}/`")
            downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=is_llamacpp)
            yield ("Done!")
    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')
    pass


def update_truncation_length(current_length, state):
    if 'loader' in state:
        if state['loader'].lower().startswith('exllama'):
            return state['max_seq_len']
        elif state['loader'] in ['llama.cpp', 'llamacpp_HF', 'ctransformers']:
            return state['n_ctx']

    return current_length


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('MODEL', type=str, default=None, nargs='?')
    parser.add_argument('--branch', type=str, default='main', help='Name of the Git branch to download from.')
    parser.add_argument('--threads', type=int, default=4, help='Number of files to download simultaneously.')
    parser.add_argument('--text-only', action='store_true', help='Only download text files (txt/json).')
    parser.add_argument('--specific-file', type=str, default=None, help='Name of the specific file to download (if not provided, downloads all).')
    parser.add_argument('--output', type=str, default=None, help='The folder where the model should be saved.')
    parser.add_argument('--clean', action='store_true', help='Does not resume the previous download.')
    parser.add_argument('--check', action='store_true', help='Validates the checksums of model files.')
    parser.add_argument('--max-retries', type=int, default=5, help='Max retries count when get error in download time.')
    args = parser.parse_args()

    branch = args.branch
    model = args.MODEL
    specific_file = args.specific_file

    if model is None:
        print("Error: Please specify the model you'd like to download (e.g. 'python download-model.py facebook/opt-1.3b').")
        sys.exit()

    downloader = ModelDownloader(max_retries=args.max_retries)
    # Clean up the model/branch names
    try:
        model, branch = downloader.sanitize_model_and_branch_names(model, branch)
    except ValueError as err_branch:
        print(f"Error: {err_branch}")
        sys.exit()

    # Get the download links from Hugging Face
    links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=args.text_only, specific_file=specific_file)

    # Get the output folder
    output_folder = downloader.get_output_folder(model, branch, is_lora, is_llamacpp=is_llamacpp, base_folder=args.output)

    if args.check:
        # Check previously downloaded files
        downloader.check_model_files(model, branch, links, sha256, output_folder)
    else:
        # Download files
        downloader.download_model_files(model, branch, links, sha256, output_folder, specific_file=specific_file, threads=args.threads, is_llamacpp=is_llamacpp)

# video chat with OpenAI models (pipe real-time emotion logs along with user's chats)

from PyQt5.QtWidgets import QApplication # GUI uses PyQt
from PyQt5.QtCore import QThread # videoplayer lives in a QThread
from gui import ChatApp, VideoPlayerWorker
from emili_core_old_with_logging import * # core threading logic

import sys
import argparse
from paz.backend.camera import Camera
import threading
import time
from datetime import datetime
import os
import requests
import subprocess
import asyncio
import functools
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from openai import OpenAI
from anthropic import Anthropic

class APIManager:
    def __init__(self, api):
        self.openai_client = OpenAI()
        self.anthropic_client = Anthropic()
        self.api = api
        self.default = "gpt-4-0125-preview"
        self.vision = "gpt-4-vision-preview"
        self.secondary = "gpt-3.5-turbo-0125"
        self.max_context_length = 16000

    def get_model(self):
        if self.api == "openai":
            self.default = "gpt-4-0125-preview"
            self.vision = "gpt-4-vision-preview"
            self.secondary = "gpt-3.5-turbo-0125"
            self.max_context_length = 16000
        elif self.api == "anthropic":
            self.default = "claude-3-5-sonnet-20240620"
            self.secondary = "claude-3-haiku-20240307"
            self.max_context_length = 16000
        elif self.api == 'ollama':
            self.default = "llama3"
            self.max_context_length = 16000
        else:
            raise ValueError(f"{self.api} API not supported. Currently supported APIs include --openai, --anthropic and --ollama")

    @retry(wait=wait_exponential(multiplier=1.5, min=1, max=60), stop=stop_after_attempt(6), retry=retry_if_exception_type(Exception))
    def claude_api_call(self, messages, model=None, temperature=1.0, max_tokens=64, seed=1331, return_full_response=False):
        if model is None:
            model = self.get_model()
        try:
            response = self.anthropic_client.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                max_tokens=max_tokens
            )
            if return_full_response:
                return response
            else:
                return response.choices[0].message.content
        except Exception as e:
            print(f"(API error: {e}, retrying...)")
            raise e

    @retry(wait=wait_exponential(multiplier=1.5, min=1, max=60), stop=stop_after_attempt(6), retry=retry_if_exception_type(Exception))
    def openai_api_call(self, messages, model=None, temperature=1.0, max_tokens=64, seed=1331, return_full_response=False):
        if model is None:
            model = self.get_model()
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                max_tokens=max_tokens
            )
            if return_full_response:
                return response
            else:
                return response.choices[0].message.content
        except Exception as e:
            print(f"(API error: {e}, retrying...)")
            raise e

    @retry(wait=wait_exponential(multiplier=1.5, min=1, max=60), stop=stop_after_attempt(6), retry=retry_if_exception_type(Exception))
    def ollama_api_call(self, messages, model=None, temperature=1.0, max_tokens=64, seed=1331, return_full_response=False):
        if model is None:
            model = self.get_model()
        try:
            url = "http://localhost:11434/api/generate"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed
            }
            response = requests.post(url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                response_data = response.json()
                if return_full_response:
                    return response_data
                else:
                    return response_data.get('response', '')
            else:
                return f"Error: {response.status_code}, {response.text}"
        except Exception as e:
            print(f"An error occurred with Ollama API: {e}")
            raise

    @functools.lru_cache(maxsize=100)
    def cached_ai_api_call(self, api: str, prompt: str, **kwargs):
        return self.ai_api_call(api, prompt, **kwargs)

    async def async_ai_api_call(self, api: str, prompt: str, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.ai_api_call, api, prompt, **kwargs)

    async def batch_ai_api_calls(self, api: str, prompts):
        tasks = [self.async_ai_api_call(api, prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

if __name__ == "__main__":
    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    transcript_path = "transcript"
    if not os.path.exists(transcript_path):
        os.makedirs(transcript_path)
    
    directory = f'{transcript_path}/{start_time_str}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    snapshot_path = "snapshot"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if use_tts:
        tts_path = "tts_audio"
        if not os.path.exists(tts_path):
            os.makedirs(tts_path)

    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-a', '--api', type=str, default='openai', help='Selected Model API')
    parser.add_argument('-c', '--camera_id', type=int, default=0, help='Camera device ID')
    parser.add_argument('-o', '--offset', type=float, default=0.1, help='Scaled offset to be added to bounding boxes')
    args = parser.parse_args()

    client = APIManager(args.api)
    camera = Camera(args.camera_id)

    model_name = client.default
    secondary_model_name = client.secondary
    vision_model_name = client.vision
    max_context_length = client.max_context_length

    chat_window_dims = [600, 600]
    app = QApplication(sys.argv)
    gui_app = ChatApp(start_time, chat_window_dims, user_chat_name, assistant_chat_name, chat_queue, chat_timestamps, new_chat_event, end_session_event)

    pipeline = Emolog(start_time, [args.offset, args.offset], f'{directory}/Emili_raw_{start_time_str}')
    user_id = 100000

    tick_thread = threading.Thread(target=tick)
    tick_thread.start()

    EMA_thread = threading.Thread(target=EMA_thread, args=(start_time, snapshot_path, pipeline), daemon=True)
    EMA_thread.start()

    sender_thread = threading.Thread(
        target=sender_thread,
        args=(model_name, vision_model_name, secondary_model_name, max_context_length, gui_app, transcript_path, start_time_str),
        daemon=True
    )
    sender_thread.start()

    assembler_thread = threading.Thread(target=assembler_thread, args=(start_time, snapshot_path, pipeline, user_id), daemon=True)
    assembler_thread.start()

    print(f"Video chat with {model_name} using emotion labels sourced from on-device camera.")
    print(f"Chat is optional, the assistant will respond to your emotions automatically!")
    print(f"Type 'q' to end the session.")

    gui_app.show()

    print("Started GUI app.")
    print("gui_app.thread()", gui_app.thread())
    print("QThread.currentThread()", QThread.currentThread())

    video_dims = [800, 450]
    video_thread = QThread()
    video_worker = VideoPlayerWorker(start_time, video_dims, pipeline, camera)
    video_worker.moveToThread(video_thread)

    video_thread.started.connect(video_worker.run)
    video_worker.finished.connect(video_thread.quit)
    video_worker.finished.connect(video_worker.deleteLater)
    video_thread.finished.connect(video_thread.deleteLater)
    video_worker.frameReady.connect(gui_app.display_frame)

    video_thread.start()
    print("Started video thread.")
    app.exec_()

    print("GUI app closed by user.")
    video_thread.quit()
    print("Video thread closed.")
    new_chat_event.set()
    assembler_thread.join()
    print("Assembler thread joined.")
    new_message_event.set()
    sender_thread.join()
    print("Sender thread joined.")
    tick_event.set()
    EMA_thread.join()
    print("EMA thread joined.")
    tick_thread.join()
    print("Tick thread joined.")

    print("Session ended.")
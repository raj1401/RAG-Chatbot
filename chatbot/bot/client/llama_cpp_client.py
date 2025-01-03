import os
from pathlib import Path
from typing import Any, Iterator

import requests
from llama_cpp import CreateCompletionResponse, CreateCompletionStreamResponse, Llama
from tqdm import tqdm

from bot.client.prompt import (
    CTX_PROMPT_TEMPLATE,
    QA_PROMPT_TEMPLATE,
    REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
    REFINED_CTX_PROMPT_TEMPLATE,
    REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
    SYSTEM_TEMPLATE,
    generate_conversation_awareness_prompt,
    generate_ctx_prompt,
    generate_qa_prompt,
    generate_refined_ctx_prompt,
)

from bot.model.base_model import ModelSettings


class LlamaCppClient:
    """
    Implementation of the Llama C++ client.
    """

    def __init__(self, model_folder: Path, model_settings: ModelSettings):
        self.model_settings = model_settings
        self.model_folder = model_folder
        self.model_path = self.model_folder / self.model_settings.file_name

        self.auto_download()
        self.llm = self.load_llm()
    
    def load_llm(self):
        llm = Llama(model_path=str(self.model_path), **self.model_settings.config)
        return llm

    def auto_download(self):
        """
        Downloads the model based on the given name and saves it
        """
        file_name = self.model_settings.file_name
        url = self.model_settings.url

        if not os.path.exists(self.model_path):
            try:
                print(f"Downloading model {file_name}...")
                response = requests.get(url, stream=True)
                with open(self.model_path, "wb") as file:
                    for chunk in tqdm(response.iter_content(chunk_size=8912)):
                        if chunk:
                            file.write(chunk)
            except Exception as e:
                print(f"Error downloading model {file_name}: {e}")
                return
        
            print(f"Model {file_name} downloaded successfully.")
        
    def generate_answer(self, prompt: str, max_new_tokens: int = 256) -> str:
        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_TEMPLATE},
                {"role": "user", "content": f"{prompt}"},
            ],
            max_tokens=max_new_tokens,
            **self.model_settings.config_answer,
        )

        answer = output["choices"][0]["message"].get("content", "")
        return answer
    
    def stream_answer(self, prompt: str, max_new_tokens: int = 256) -> str:
        answer = ""
        stream = self.start_answer_iterator_streamer(prompt, max_new_tokens=max_new_tokens)

        for output in stream:
            token = output["choices"][0]["delta"].get("content", "")
            answer += token
            print(token, end="", flush=True)
        
        return answer
    
    def start_answer_iterator_streamer(self, prompt: str, max_new_tokens: int = 256) -> CreateCompletionResponse | Iterator[CreateCompletionStreamResponse]:
        stream = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_TEMPLATE},
                {"role": "user", "content": f"{prompt}"},
            ],
            max_tokens=max_new_tokens,
            stream=True,
            **self.model_settings.config_answer,
        )

        return stream
    
    @staticmethod
    def parse_token(token):
        return token["choices"][0]["delta"].get("content", "")

    @staticmethod
    def generate_qa_prompt(question: str) -> str:
        return generate_qa_prompt(QA_PROMPT_TEMPLATE, SYSTEM_TEMPLATE, question)

    @staticmethod
    def generate_ctx_prompt(question: str, context: str) -> str:
        return generate_ctx_prompt(CTX_PROMPT_TEMPLATE, SYSTEM_TEMPLATE, context, question)
    
    @staticmethod
    def generate_refined_ctx_prompt(question: str, context: str, existing_answer: str) -> str:
        return generate_refined_ctx_prompt(REFINED_CTX_PROMPT_TEMPLATE, SYSTEM_TEMPLATE, question, existing_answer, context)
    
    @staticmethod
    def generate_refined_question_conversation_awareness_prompt(question: str, chat_history: str) -> str:
        return generate_conversation_awareness_prompt(REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE, SYSTEM_TEMPLATE, question, chat_history)
    
    @staticmethod
    def generate_refined_answer_conversation_awareness_prompt(question: str, chat_history: str) -> str:
        return generate_conversation_awareness_prompt(REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE, SYSTEM_TEMPLATE, question, chat_history)

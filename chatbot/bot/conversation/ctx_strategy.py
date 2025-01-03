from enum import Enum
from typing import Any

from helpers.log import get_logger
from entities.document import Document
from bot.client.llama_cpp_client import LlamaCppClient


logger = get_logger(__name__)

class CreateAndRefineStrategy:
    """
    Class for sequential refinement of responses using retrieved contents.
    """

    def __init__(self, llm: LlamaCppClient):
        self.llm = llm
    
    def generate_response(self, retrived_contents: list[Document], question: str, max_new_tokens: int = 256) -> str | Any:
        """
        Generates response by refining the original answer with the retrieved contents.
        We start with the first content and generate an initial response using the contextual prompt.
        Then, for subsequent contents, we refine the response using additional context using the refined contextual prompt.

        This function returns a response generator.
        """
        current_response = None
        fmt_prompts = []

        if not retrived_contents:
            qa_prompt = self.llm.generate_qa_prompt(question=question)
            logger.info("--- Generating a single response ... ---")
            response = self.llm.start_answer_iterator_streamer(prompt=qa_prompt, max_new_tokens=max_new_tokens)
            return response, qa_prompt

        num_contents = len(retrived_contents)

        for idx, node in enumerate(retrived_contents, start=1):
            logger.info(f"--- Generating an answer for the chunk {idx} ... ---")
            context = node.page_content
            logger.debug(f"--- Context: '{context}' ... ---")
            if idx == 0:
                fmt_prompt = self.llm.generate_ctx_prompt(question=question, context=context)
            else:
                fmt_prompt = self.llm.generate_refined_ctx_prompt(question=question, existing_answer=str(current_response), context=context)
            
            if idx == num_contents:
                current_response = self.llm.start_answer_iterator_streamer(fmt_prompt, max_new_tokens=max_new_tokens)
            else:
                current_response = self.llm.generate_answer(fmt_prompt, max_new_tokens=max_new_tokens)
                logger.debug(f"--- Current response: '{current_response}' ... ---")
            
            fmt_prompts.append(fmt_prompt)
        
        return current_response, fmt_prompts
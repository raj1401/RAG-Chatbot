from typing import Any

from helpers.log import get_logger
from entities.document import Document
from bot.client.llama_cpp_client import LlamaCppClient
from bot.conversation.ctx_strategy import CreateAndRefineStrategy


logger = get_logger(__name__)

class ConversationRetrieval:
    """
    Class that manages conversation retrieval.
    """

    def __init__(self, llm: LlamaCppClient) -> None:
        self.llm = llm
        self.chat_history = []
    
    def get_chat_history(self) -> list[tuple[str, str]]:
        return self.chat_history
    
    def update_chat_history(self, question: str, answer: str) -> list[tuple[str, str]]:
        self.chat_history.append((question, answer))
        self.chat_history = self.keep_chat_history_size()
        return self.chat_history
    
    def keep_chat_history_size(self, max_size: int = 2) -> list[tuple[str, str]]:
        if len(self.chat_history) > max_size:
            self.chat_history = self.chat_history[-max_size:]
        return self.chat_history
    
    def refine_question(self, question: str, max_new_tokens: int = 256) -> str:
        """
        Refines the question based on the chat history.
        """
        if self.get_chat_history():
            questions_and_answers = [
                "\n".join([f"question: {qa[0]}", f"answer: {qa[1]}"]) for qa in self.get_chat_history()
            ]
            chat_history = "\n".join(questions_and_answers)

            logger.info("--- Refining the question using the chat history... ---")

            conversation_awareness_prompt = self.llm.generate_refined_question_conversation_awareness_prompt(
                question=question, chat_history=chat_history
            )

            logger.info(f"--- Prompt:\n {conversation_awareness_prompt} \n---")
            refined_question = self.llm.generate_answer(conversation_awareness_prompt, max_new_tokens=max_new_tokens)
            logger.info(f"--- Refined Question: {refined_question} ---")

            return refined_question
        else:
            return question
        
    def answer(self, question: str, max_new_tokens: int = 256) -> str:
        """
        Generates an answer to the question based on given prompt or chat history.
        First, we check if there is an existing chat history. If there is, we construct
        a conversation awareness prompt which is then used to generate the answer using an LLM.
        If there is no chat history, we simply generate an answer using a prompt constructed
        from the question.
        """
        if self.get_chat_history():
            questions_and_answers = [
                "\n".join([f"question: {qa[0]}", f"answer: {qa[1]}"]) for qa in self.get_chat_history()
            ]
            chat_history = "\n".join(questions_and_answers)

            logger.info("--- Answer the question based on the chat history... ---")

            conversation_awareness_prompt = self.llm.generate_refined_answer_conversation_awareness_prompt(
                question=question, chat_history=chat_history
            )

            logger.debug(f"--- Prompt:\n {conversation_awareness_prompt} \n---")

            streamer = self.llm.start_answer_iterator_streamer(conversation_awareness_prompt, max_new_tokens=max_new_tokens)
            return streamer
        else:
            prompt = self.llm.generate_qa_prompt(question=question)
            logger.debug(f"--- Prompt:\n {prompt} \n---")
            streamer = self.llm.start_answer_iterator_streamer(prompt, max_new_tokens=max_new_tokens)
            return streamer
    
    @staticmethod
    def context_aware_answer(
        ctx_synthesis_strategy: CreateAndRefineStrategy, question: str,
        retrived_contents: list[Document], max_new_tokens: int = 256
    ):
        """
        Generates an answer to the question based on the retrieved contents.
        """
        streamer, fmt_prompts = ctx_synthesis_strategy.generate_response(
            retrived_contents=retrived_contents, question=question, max_new_tokens=max_new_tokens
        )
        return streamer, fmt_prompts
        
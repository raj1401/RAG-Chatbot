# Templates for the assistant

SYSTEM_TEMPLATE = """You are a helpful, respectful and honest assistant."""


# Basic Question Answering Template
QA_PROMPT_TEMPLATE = """Answer the question below:
{question}
"""
def generate_qa_prompt(template: str, system: str, question: str) -> str:
    prompt = template.format(system=system, question=question)
    return prompt


# Refined Conversation Awareness Question Answering Template
REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE = """Chat History:
---------------------
{chat_history}
---------------------
Follow Up Question: {question}
Given the above conversation and a follow up question, rephrase the follow up question to be a standalone question.
Standalone question:
"""

REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE = """
You are engaging in a conversation with a human participant who is unaware that they might be
interacting with a machine.
Your goal is to respond in a way that convincingly simulates human-like intelligence and behavior.
The conversation should be natural, coherent, and contextually relevant.
Chat History:
---------------------
{chat_history}
---------------------
Follow Up Question: {question}\n
Given the context provided in the Chat History and the follow up question, please answer the follow up question above.
If the follow up question isn't correlated to the context provided in the Chat History, please just answer the follow up
question, ignoring the context provided in the Chat History.
Please also don't reformulate the follow up question, and write just a concise answer.
"""
def generate_conversation_awareness_prompt(template: str, system: str, question: str, chat_history: str) -> str:
    prompt = template.format(
        system=system,
        chat_history=chat_history,
        question=question,
    )
    return prompt


# Contextual Question Answering Template
CTX_PROMPT_TEMPLATE = """Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question below:
{question}
"""
def generate_ctx_prompt(template: str, system: str, context: str, question: str) -> str:
    prompt = template.format(system=system, context=context, question=question)
    return prompt


# Refined Contextual Question Answering Template
REFINED_CTX_PROMPT_TEMPLATE = """The original query is as follows: {question}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer
(only if needed) with some more context below.
---------------------
{context}
---------------------
Given the new context, refine the original answer to better answer the query.
If the context isn't useful, return the original answer.
Refined Answer:
"""
def generate_refined_ctx_prompt(template: str, system: str, question: str, existing_answer: str, context: str = "") -> str:
    prompt = template.format(
        system=system,
        context=context,
        existing_answer=existing_answer,
        question=question,
    )
    return prompt

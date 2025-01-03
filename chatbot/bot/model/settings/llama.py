from bot.model.base_model import ModelSettings


"""
n_ctx - The max sequence length to use - note that longer sequence lengths require much more resources
n_threads - The number of CPU threads to use, tailor to your system and the resulting performance
n_gpu_layers - The number of layers to offload to GPU, if you have GPU acceleration available
"""

class Llama31Settings(ModelSettings):
    url = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    file_name = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    config = {
        "n_ctx": 4096,
        "n_threads": 8,
        "n_gpu_layers": 50,
    }
    config_answer = {"temperature": 0.7, "stop": []}


class Llama32OneSettings(ModelSettings):
    url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_M.gguf"
    file_name = "Llama-3.2-1B-Instruct-Q5_K_M.gguf"
    config = {
        "n_ctx": 4096,
        "n_threads": 8,
        "n_gpu_layers": 50,
    }
    config_answer = {"temperature": 0.7, "stop": []}


class Llama32ThreeSettings(ModelSettings):
    # There is also the uncensored version: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF
    url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q5_K_M.gguf"
    file_name = "Llama-3.2-3B-Instruct-Q5_K_M.gguf"
    config = {
        "n_ctx": 4096,
        "n_threads": 8,
        "n_gpu_layers": 50,
    }
    config_answer = {"temperature": 0.7, "stop": []}

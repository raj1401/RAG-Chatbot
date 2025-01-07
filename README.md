# Installation Instructions

Follow these steps to set up and run the chatbot application:

### Step 1: Create the Conda Environment
Run the following command to create a new Conda environment:
```bash
conda env create -f environment.yml
```

### Step 2: Activate the Environment
Activate the newly created environment:
```bash
conda activate rag-chatbot
```

### Step 3: Install `llama-cpp-python`
Install version `0.2.85` of `llama-cpp-python` using platform-specific instructions from the [llama-cpp-python GitHub repository](https://github.com/abetlen/llama-cpp-python).

### Step 4: Install HuggingFace Hub
Install version `0.25.0` of HuggingFace Hub:
```bash
pip install huggingface_hub==0.25.0
```

### Step 5: Prepare the Model
1. Create a `models` directory in your project folder.
2. Download the GGUF file for your desired model. (Refer to the links in `bot/model/settings/llama.py` for model options.)

### Step 6: Prepare Documentation
1. Create a `docs` directory.
2. Place your Markdown files into the `docs` directory.

### Step 7: Build the Memory
Run the following script to process and chunk your documentation files:
```bash
python chatbot/memory_builder.py --chunk-size {CHUNK_SIZE} --chunk-overlap {CHUNK_OVERLAP}
```
Replace `{CHUNK_SIZE}` and `{CHUNK_OVERLAP}` with appropriate values.

### Step 8: Run the Chatbot Application
Start the chatbot application using Streamlit:
```bash
streamlit run chatbot/app.py --model {MODEL_NAME} --k {K}
```
Replace `{MODEL_NAME}` with the name of your chosen model and `{K}` with the desired number of results to retrieve.


(This project was inspired by @umbertogriffo's implementation of an RAG ChatBot, with the goal of keeping the implementation simple and straightforward.)
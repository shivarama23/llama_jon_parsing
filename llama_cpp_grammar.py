import time
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
from llama_cpp.llama import Llama, LlamaGrammar
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

grammar = LlamaGrammar.from_file("./grammar.gbnf")

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 1024  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    # streamming=True,
    model_path="./llama3-3b/Llama-3.2-3B-Instruct-Q8_0.gguf",
    # model_path="./llama2-7b/llama-2-7b-chat.Q4_K_M.gguf",
    n_gpu_layers=6,
    n_batch=n_batch,
    f16_kv=True,
    temperature=1,
    max_tokens=-1,
    # callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
    grammar_path="./grammar.gbnf"
)

from mha_test import prompt, text_list
import json

errored_indices = []
no_of_errors = 0

# for i, text in enumerate(text_list):
text = text_list[0]
response = llm.invoke(prompt+'\n'+text)

print(response)

try:
    response_json = json.loads(response)
except json.JSONDecodeError as e:
    # errored_indices.append(i)
    no_of_errors += 1
    print(f"Failed to parse response as JSON: {e}")
    # continue

print(f"Failed to parse {no_of_errors} responses as JSON: {errored_indices}")
import json
from enum import Enum
from pathlib import Path

import ollama

import pandas as pd
from IPython.display import Image, Markdown, display
from tqdm import tqdm

from mha_test import text_list

MODEL = "llama3"
TEMPERATURE = 0

class ResponseFormat(Enum):
    JSON = "json_object"
    TEXT = "text"


def call_model(
    prompt: str, response_format: ResponseFormat = ResponseFormat.TEXT
) -> str:
    response = ollama.generate(
        model=MODEL,
        prompt=prompt,
        keep_alive="1h",
        format="" if response_format == ResponseFormat.TEXT else "json",
        options={"temperature": TEMPERATURE},
    )
    return response["response"]


prompt = """
You are an AI assistant which can create structured profiles based on paragraphs of information about individuals. You are required to extract the following information from the given text:

"DESIGNATIONS": The designations held by the individual, along with the location and the duration of the tenure. The format should be "Designation _ location _ Start Year - End Year". If the information is not available, return 9999.
"FIRST-EVER": Mention sentences specifying first-ever facts done for the first time EVER in the WORLD not done similar to this by any other person. Else return 9999."CAREER_DURATION": Return Longevity: total number of years worked in the field. numeric answer, without sentences, add . at the end of year if such mentioned in raw text, if not found return 9999.

The expected output is a JSON object. The JSON object should be of the following format:
{
  "DESIGNATIONS": [ all the designations held]
  "FIRST-EVER": [ all mentions of first-ever facts]
  "CAREER-DURATION": []
}
Always generate complete json output only. COver all the pointers mentioned above. DO NOT include any additional information in the output. Do not miss any facts mentioned in the input text.
"""

errored_indices = []
total_errors = 0

for i, text in tqdm(enumerate(text_list)):
    response = call_model(prompt + "\n" + text, response_format=ResponseFormat.JSON)
    try:
        response_json = json.loads(response)
    except json.JSONDecodeError as e:
        errored_indices.append(i)
        total_errors += 1
        print(f"Failed to parse response as JSON: {e}")
        continue


print(f"Failed to parse {total_errors} responses as JSON: {errored_indices}")
print(f"Successflly parsed {len(text_list) - total_errors} responses as JSON")
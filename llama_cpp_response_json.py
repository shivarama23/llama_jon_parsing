import json
from llama_cpp import Llama
from mha_test import text_list

llm = Llama(
      model_path="./llama3-3b/Llama-3.2-3B-Instruct-Q8_0.gguf",
      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      n_ctx=2048, # Uncomment to increase the context window
)


prompt = """
You are an AI assistant which can create structured profiles based on paragraphs of information about individuals. You are required to extract the following information from the given text:

"DESIGNATIONS": The designations and work related posts held by the individual, along with the location and the duration of the tenure. The format should be "Designation _ location _ Start Year - End Year". If the information is not available, return 9999.
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
for i, text in enumerate(text_list):
    output = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs in JSON.",
            },
            {"role": "user", "content": prompt+'\n'+text},
        ],
        response_format={
            "type": "json_object",
        },
        temperature=0.7,
    )
    print(output)
    try:
        response = json.loads(output["choices"][0]["message"]["content"])
    except json.JSONDecodeError as e:
        print(f"Failed to parse response as JSON: {e}")
        errored_indices.append(i)
        total_errors += 1
        continue

print(f"Failed to parse {total_errors} responses as JSON: {errored_indices}")
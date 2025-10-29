from openai import OpenAI
import base64
import os

from Prompts.prompt_system import system_prompt
from API_keys.keys import OPENAI_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def init_gpt4():
    client = OpenAI()
    return client

def get_response(client, prompt, image_path, model_name="gpt-4o", max_tokens=512): # "gpt-4-turbo", "gpt-4o"
    base64_image = encode_image(image_path)
    messages = [
                    {
                        'role': 'system',
                        'content': system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                          },
                    ],
                    }
              ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
    )

    out = response.choices[0].message.content
    return out

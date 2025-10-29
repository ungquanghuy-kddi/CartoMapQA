import anthropic
import os
import base64
from Prompts.prompt_system import system_prompt
from API_keys.keys import CLAUDE_KEY

os.environ["ANTHROPIC_API_KEY"] = CLAUDE_KEY    


def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode('utf-8')
        return base64_string

def init_claude_model():
    client = anthropic.Anthropic()
    return client

def get_response(client, prompt, image_path, model_name="claude-3-7-sonnet-20250219",
                 thinking=True):
    message_list = [
        {
            "role": 'user',
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": get_base64_encoded_image(image_path)}},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    if thinking:
        message = client.messages.create(
            model=model_name,
            max_tokens=1025,
            thinking={
            "type": "enabled",
            "budget_tokens": 1024
            },
            system=system_prompt,
            messages=message_list,
        )
        if len(message.content) > 1:
            out = message.content[1].text
            thinking = message.content[0].thinking
        else:
            out = ""
            thinking = message.content[0].thinking
        return out, thinking
    else:
        message = client.messages.create(
            model=model_name,
            max_tokens=1025,
            system=system_prompt,
            messages=message_list,
        )
        out = message.content[0].text
    return out
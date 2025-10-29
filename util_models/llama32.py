from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

from Prompts.prompt_system import system_prompt


def init_llama_vision(model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"):
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def get_response(model, prompt, image_path, processor, max_tokens=512):
    image = Image.open(image_path)

    messages = [
    {"role": "system", "content": {"type": "text", "text": system_prompt}},
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt}
    ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=max_tokens)
    return processor.decode(output[0]).split("assistant<|end_header_id|>")[1].split("<|eot_id|>")[0]
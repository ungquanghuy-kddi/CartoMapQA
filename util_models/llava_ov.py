from PIL import Image
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from Prompts.prompt_system import system_prompt

def init_llava_ov(model_id = "llava-hf/llava-onevision-qwen2-72b-ov-hf"):
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def get_response(model, processor, prompt, image_path, max_tokens=512):
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    conversation = [
        {"role": "system", "content": {"type": "text", "text": system_prompt}},
        {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    raw_image = Image.open(image_path)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(model.device, torch.bfloat16)

    output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    response = processor.decode(output[0], skip_special_tokens=False)
    if ">assistant" in response:
        response = response.split(">assistant")[1].replace("<|im_end|>", "").strip()
    return response
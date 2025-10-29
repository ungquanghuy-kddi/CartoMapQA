from util_models import llama32, llama4_scout, llava_ov, gpt4, gemini, openai_o_series, claude_sonnet
from util_models import  internvl25, qwenvl25

from Parse.parse_response_stmf import parse_stmf
from Prompts.prompt_stmf import prompt_stmf
from Prompts.prompt_system import system_prompt

import os
import json


def get_response_stmf(model_name, question_type, map_path="./Dataset/CartoMapQA/MapFeatureUnderstanding/_Maps", 
                        question_path="./Dataset/CartoMapQA/MapFeatureUnderstanding/STMF_task/{}.json",
                        save_path="./Responses/STMF"):
    
    # initialize model
    if model_name == "Llama-3.2-11B":
        model, processor = llama32.init_llama_vision(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct")
    elif model_name == "Llama-3.2-90B":
        model, processor = llama32.init_llama_vision(model_id="meta-llama/Llama-3.2-90B-Vision-Instruct")
    elif model_name == "Llama-4-Scout":
        model, processor = llama4_scout.init_llama4_scout(model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct")
    
    elif model_name == "LlaVA-OV-7B":
        model, processor = llava_ov.init_llava_ov(model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf")
    elif model_name == "LlaVA-OV-72B":
        model, processor = llava_ov.init_llava_ov(model_id = "llava-hf/llava-onevision-qwen2-72b-ov-hf")
    
    elif model_name == "InternVL-2.5-8B":
        model, tokenizer = internvl25.init_internvl2_5(model_id = "OpenGVLab/InternVL2_5-8B")
    elif model_name == "InternVL-2.5-78B":
        model, tokenizer = internvl25.init_internvl2_5(model_id = "OpenGVLab/InternVL2_5-78B")
    
    elif model_name == "Qwen-VL-2.5-7B":
        model, processor = qwenvl25.init_qwenvl(model_id="Qwen/Qwen2.5-VL-7B-Instruct")
    elif model_name == "Qwen-VL-2.5-72B":
        model, processor = qwenvl25.init_qwenvl(model_id="Qwen/Qwen2.5-VL-72B-Instruct")

    elif model_name == "gpt-4-turbo-2024-04-09" or model_name == "gpt-4o-2024-08-06":
        client = gpt4.init_gpt4()

    elif model_name == "o1-2024-12-17" or model_name == "o3-2025-04-16":
        client = openai_o_series.init_openai_o_series()
    
    elif model_name == "gemini-2.5-pro-preview-03-25":
        model = gemini.init_gemini(model_name)
        
    elif model_name == "claude-3-7-sonnet-20250219":
        model = claude_sonnet.init_gemini(model_name)
    else:
        print ("The model's name does not exist in our evaluation. Please check the list of model's name")
        exit(0)

    with open (question_path.format(question_type)) as f:
        data = json.load(f)

    for item in data:
        question_id = item.split("_")[1]
        question = data[item]
        fn = question['Image']
        image_path = os.path.join(map_path, question['City'], question['Image'])
        prompt = prompt_stmf(question_type, question['MF type'])

        # get response from the model
        if model_name == "Llama-3.2-11B" or model_name == "Llama-3.2-90B":
            response = llama32.get_response(model, prompt, image_path, processor, max_tokens=512)

        elif model_name == "Llama-4-Scout":
            response = llama4_scout.get_response(model, prompt, image_path, processor, max_tokens=512)

        elif model_name == "LlaVA-OV-7B" or model_name == "LlaVA-OV-72B":
            response = llava_ov.get_response(model, processor, prompt, image_path, max_tokens=512)

        elif model_name == "InternVL-2.5-8B" or model_name == "InternVL-2.5-78B":
            prompt = system_prompt + "\n" + prompt # add system prompt, other models are already added to each request
            image_paths = [image_path]
            response = internvl25.get_response(model, prompt, image_paths, tokenizer, max_tokens=512)
        
        elif model_name == "Qwen-VL-2.5-7B" or model_name == "Qwen-VL-2.5-72B":
            response = qwenvl25.get_response(model, prompt, image_path, processor, max_tokens=512)

        elif model_name == "gpt-4-turbo-2024-04-09" or model_name == "gpt-4o-2024-08-06":
            response = gpt4.get_response(client, prompt, image_path, model_name, max_tokens=512)

        elif model_name == "o1-2024-12-17" or model_name == "o3-2025-04-16":
            response = openai_o_series.get_response(client, prompt, image_path, model_name)

        elif model_name == "gemini-2.5-pro-preview-03-25":
            prompt = system_prompt + "\n" + prompt # add system prompt, other models are already added to each request
            response = gemini.get_response(model, prompt, image_path)
        
        elif model_name == "claude-3-7-sonnet-20250219":
            response, thinking = claude_sonnet.get_response(client, prompt, image_path, 
                                                  model_name, thinking=True)
        
        answer = parse_stmf(response, question_type, fn)

        # save the answer and response to json files
        new_data = question
        new_data['Response'] = response
        new_data['Answer'] = answer

        if model_name == "claude-3-7-sonnet-20250219":
            new_data[item]['Thinking'] = thinking

        save_fn = question["City"] + "_" + fn[:-4] + "_" + f"q{question_id}_" + f"{question_type}.json"
        path_out = os.path.join(save_path, model_name, question_type)
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        with open(os.path.join(path_out, save_fn), 'w') as f:
            json_object = json.dumps(new_data, indent=4)
            f.write(json_object)

    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Get responses for STMF task")
    parser.add_argument('--model_name', type=str, default="gpt-4o-2024-08-06", help="Name of model to be evaluated")
    parser.add_argument('--question_type', type=str, required=True, choices=["presence", "counting", "name_listing"],
                        help="Type of question to be answered")
    parser.add_argument('--map_path', type=str, default="./Dataset/CartoMapQA/MapFeatureUnderstanding/_Maps",
                        help="Path to the map images")
    parser.add_argument('--question_path', type=str, default="./Dataset/CartoMapQA/MapFeatureUnderstanding/STMF_task/{}.json",
                        help="Path to the question JSON file")
    parser.add_argument('--save_path', type=str, default="./Responses/STMF",
                        help="Directory to save the responses")

    args = parser.parse_args()

    get_response_stmf(args.model_name, args.question_type, args.map_path, args.question_path, args.save_path)
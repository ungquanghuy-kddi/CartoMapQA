from util_models import llama32, llama4_scout, llava_ov, gpt4, gemini, openai_o_series, claude_sonnet
from util_models import internvl25, qwenvl25

from Parse.parse_response_mtmf import parse_mtmf
from Prompts.prompt_mtmf import prompt_mtmf
from Prompts.prompt_system import system_prompt

import os
import json
import copy

from utils import extract_json_from_file

def get_response_mtmf(model_name, map_path="./Dataset/CartoMapQA/MapFeatureUnderstanding/_Maps", 
                        groundtruth_path="./Dataset/CartoMapQA/MapFeatureUnderstanding/MTMF_task/Groundtruth",
                        save_path="./Responses/MTMF"):
    
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

    areas = os.listdir(map_path)
    for area in areas:
        for fn in os.listdir(os.path.join(map_path, area)):    
            image_path = os.path.join(map_path, area, fn)
            prompt = prompt_mtmf()

            # get response from the model
            if model_name == "Llama-3.2-11B" or model_name == "Llama-3.2-90B":
                response = llama32.get_response(model, prompt, image_path, processor, max_tokens=1024)

            elif model_name == "Llama-4-Scout":
                response = llama4_scout.get_response(model, prompt, image_path, processor, max_tokens=1024)

            elif model_name == "LlaVA-OV-7B" or model_name == "LlaVA-OV-72B":
                response = llava_ov.get_response(model, processor, prompt, image_path, max_tokens=1024)

            elif model_name == "InternVL-2.5-8B" or model_name == "InternVL-2.5-78B":
                prompt = system_prompt + "\n" + prompt # add system prompt, other models are already added to each request
                image_paths = [image_path]
                response = internvl25.get_response(model, prompt, image_paths, tokenizer, max_tokens=1024)
            
            elif model_name == "Qwen-VL-2.5-7B" or model_name == "Qwen-VL-2.5-72B":
                response = qwenvl25.get_response(model, prompt, image_path, processor, max_tokens=1024)

            elif model_name == "gpt-4-turbo-2024-04-09" or model_name == "gpt-4o-2024-08-06":
                response = gpt4.get_response(client, prompt, image_path, model_name, max_tokens=1024)

            elif model_name == "o1-2024-12-17" or model_name == "o3-2025-04-16":
                response = openai_o_series.get_response(client, prompt, image_path, model_name)

            elif model_name == "gemini-2.5-pro-preview-03-25":
                prompt = system_prompt + "\n" + prompt # add system prompt, other models are already added to each request
                response = gemini.get_response(model, prompt, image_path)
            
            elif model_name == "claude-3-7-sonnet-20250219":
                response, thinking = claude_sonnet.get_response(client, prompt, image_path, 
                                                    model_name, thinking=True)
            
            try:
                response = response.replace('```json', '').replace('```', '').replace('<|eot|>', '')
                response_json = extract_json_from_file(response)
                response_json = json.loads(response)
                origin_response =  copy.deepcopy(response_json)
                
                norm_response = parse_mtmf(response_json)
            
                gt_path = os.path.join(groundtruth_path, area, fn[:-4] + ".json")
                with open(gt_path, 'r') as f:
                    groundtruth = json.load(f)

                # combine groundtruth and response
                for item in norm_response:
                    norm_response[item]["true_count"] = groundtruth[item]["count"]
                    norm_response[item]["true_names"] = groundtruth[item]["names"]

                if model_name == "claude-3-7-sonnet-20250219":
                    thinking = {"thinking": thinking}
                    out_save = [norm_response, origin_response, thinking]
                else:
                    out_save = [norm_response, origin_response]

                save_fn = area + "_" + fn[:-4] + ".json"
                path_out = os.path.join(save_path, model_name)
                if not os.path.exists(path_out):
                    os.makedirs(path_out)

                with open(os.path.join(path_out, save_fn), 'w') as f:
                    json_object = json.dumps(out_save, indent=4)
                    f.write(json_object)
            except:
                # Sometimes the model's response is not in the correct format
                # So we save the raw response
                save_fn = area + "_" + fn[:-4] + ".txt"
                path_out = os.path.join(save_path, model_name)
                if not os.path.exists(path_out):
                    os.makedirs(path_out)

                with open(os.path.join(path_out, save_fn), 'w') as f:
                    f.write(response)
            
    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="o3-2025-04-16", help="Name of model to be evaluated")
    parser.add_argument('--map_path', type=str, default="./Dataset/CartoMapQA/MapFeatureUnderstanding/_Maps", help="Path to the map images")
    parser.add_argument('--groundtruth_path', type=str, default="./Dataset/CartoMapQA/MapFeatureUnderstanding/MTMF_task/Groundtruth", help="Path to the groundtruth data")
    parser.add_argument('--save_path', type=str, default="./Responses/MTMF", help="Directory to save the responses")

    args = parser.parse_args()

    get_response_mtmf(args.model_name, args.map_path, args.groundtruth_path, args.save_path)
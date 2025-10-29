from util_models import llama32, llama4_scout, llava_ov, gpt4, gemini, openai_o_series, claude_sonnet
from util_models import internvl25, qwenvl25

from Parse.parse_response_rle import parse_rle
from Prompts.prompt_rle import prompt_rle
from Prompts.prompt_system import system_prompt
import os
import json


def get_response_rle(model_name, map_path="./Dataset/CartoMapQA/MapScaleUnderstanding/_Route", 
                        grountruth_path="./Dataset/CartoMapQA/MapScaleUnderstanding/Scale_understanding.json",
                        save_path="./Responses/RLE",
                        use_CoT=True):

    measure_objective = ["meters", "feet"]

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

    with open (grountruth_path) as f:
        data = json.load(f)

    for id, route in enumerate(data):
        for measure_unit in measure_objective:
            image_path = os.path.join(map_path, "route_" + route["route_ID"] + ".png")
            prompt = prompt_rle(measure_unit, use_CoT=use_CoT)

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
            
            answer = parse_rle(response)

            # save the answer and response to json files
            out_dict = {"Correct_answer": round(route[f"Ans_{measure_unit}"], 3),
                        "Response": response,
                        "Result": answer,
                        "Measure": measure_unit,
                        "Difficulty": route["difficulty"],
                        "City": route["city"],
                        "Zoom_level": route["zoom_scale"]}
            
            if model_name == "claude-3-7-sonnet-20250219":
                out_dict["Thinking"] = thinking
                
            path_out = os.path.join(save_path, model_name)
            if not os.path.exists(path_out):
                os.makedirs(path_out)

            difficulty = route["difficulty"]
            save_fn = "route_" + route["route_ID"] + f"_{measure_unit}_" + f"{difficulty}.json"
            with open(os.path.join(path_out, save_fn), 'w') as f:
                json_object = json.dumps(out_dict, indent=4)
                f.write(json_object)

    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="gpt-4o-2024-08-06", help="Name of model to be evaluated")
    parser.add_argument('--map_path', type=str, default="./Dataset/CartoMapQA/MapScaleUnderstanding/_Route", help="Path to the map images")
    parser.add_argument('--groundtruth_path', type=str, default="./Dataset/CartoMapQA/MapScaleUnderstanding/Scale_understanding.json", help="Path to the ground truth data")
    parser.add_argument('--save_path', type=str, default="./Responses/RLE", help="Directory to save the responses")

    args = parser.parse_args()

    get_response_rle(model_name=args.model_name, map_path=args.map_path, 
                     grountruth_path=args.groundtruth_path, save_path=args.save_path)
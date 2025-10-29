from argparse import ArgumentParser
from get_response_mfs import get_response_mfs
from get_response_stmf import get_response_stmf
from get_response_mtmf import get_response_mtmf
from get_response_rle import get_response_rle
from get_response_mml import get_response_mml
from get_response_srn import get_response_srn

import os

"""
List of support models:
    Llama-3.2-11B
    Llama-3.2-90B
    Llama-4-Scout
    LlaVA-OV-7B
    LlaVA-OV-72B
    InternVL-2.5-8B
    InternVL-2.5-78B
    Qwen-VL-2.5-7B
    Qwen-VL-2.5-72B
    
    gpt-4-turbo-2024-04-09
    gpt-4o-2024-08-06
    o1-2024-12-17
    o3-2025-04-16
    gemini-2.5-pro-preview-03-25
    claude-3-7-sonnet-20250219
"""

EXECUTE_TASKS = {"MFS": get_response_mfs, 
                 "STMF": get_response_stmf, 
                 "MTMF": get_response_mtmf, 
                 "RLE": get_response_rle,
                 "SRN": get_response_srn,
                 "MML": get_response_mml}

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default="gpt-4o-2024-08-06", help="Name of model to be evaluated")
    parser.add_argument('--save_dir', type=str, default="./Responses", help="Directory of response for evaluation")

    args = parser.parse_args()

    for task in EXECUTE_TASKS:
        save_path = os.path.join(args.save_dir, task)
        if task == "STMF":
            get_response_stmf(model_name=args.model_name, save_path=save_path, question_type="presence")
            get_response_stmf(model_name=args.model_name, save_path=save_path, question_type="counting")
            get_response_stmf(model_name=args.model_name, save_path=save_path, question_type="name_listing")
        else:            
            EXECUTE_TASKS[task](model_name=args.model_name, save_path=save_path)

if __name__ == '__main__':
    main()
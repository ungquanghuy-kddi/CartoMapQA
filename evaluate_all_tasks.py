from argparse import ArgumentParser
from evaluate_rle import evaluate_rle
from evaluate_mfs import evaluate_mfs
from evaluate_stmf import evaluate_stmf
from evaluate_mtmf import evaluate_mtmf
from evaluate_mml import evaluate_mml
from evaluate_srn import evaluate_srn

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

EVAL_TASKS = {"MFS": evaluate_mfs, 
              "STMF": evaluate_stmf, 
              "MTMF": evaluate_mtmf, 
              "RLE": evaluate_rle,
              "SRN": evaluate_srn,
              "MML": evaluate_mml}

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default="o3-2025-04-16", help="Name of model to be evaluated")
    parser.add_argument('--eval_dir', type=str, default="./Responses", help="Directory of response for evaluation")

    args = parser.parse_args()

    for task in EVAL_TASKS:
        eval_dir = os.path.join(args.eval_dir, task, args.model_name)
        EVAL_TASKS[task](args.model_name, eval_dir)

if __name__ == '__main__':
    main()
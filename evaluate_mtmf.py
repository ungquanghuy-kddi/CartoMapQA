import os
import json
import numpy as np

from eval import RMSE, Precision_name_listing, Recall_name_listing, F1_score_name_listing, R2_score
from Prompts.prompt_mtmf import POI_LIST


def evaluate_mtmf(model_name="o1-2024-12-17", eval_dir="./Responses/MTMF/o1-2024-12-17"):
    avg_res_count = [[], []] # rmse, r2
    avg_res_names = [[], [], []]  # mprec, mrec, mf1

    for id, fn in enumerate(os.listdir(eval_dir)):
        res_count = [[], []] # true, pred
        res_names = [[], []] # true, pred
        with open(os.path.join(eval_dir, fn), 'r') as f:
            data = json.load(f)[0]
        
        for k, item in POI_LIST.items():
            res_count[0].append(data[item]["true_count"])
            res_count[1].append(data[item]["count"])

            valid_name_list = [name for name in data[item]["names"] if name.strip() != ""]

            res_names[0] += data[item]["true_names"]
            res_names[1] += valid_name_list

        ### Count POI ###
        rmse = RMSE(res_count[0], res_count[1])
        r2 = R2_score(res_count[0], res_count[1])

        avg_res_count[0].append(rmse)
        avg_res_count[1].append(r2)

        ### Name listing ###
        prec = Precision_name_listing(res_names[0], res_names[1])
        rec = Recall_name_listing(res_names[0], res_names[1])
        f1 = F1_score_name_listing(prec, rec)

        avg_res_names[0].append(prec)
        avg_res_names[1].append(rec)
        avg_res_names[2].append(f1)

    print ("============== MTMF task =================")
    ### Eval counting question ###
    rmse = np.mean(avg_res_count[0])
    r2 = np.mean(avg_res_count[1])
    print ("{}\tCounting: RMSE = {:0.3f}, R2 = {:0.3f}".format(model_name, rmse, r2))

    ### Eval name-listing question ###
    avg_prec = np.mean(avg_res_names[0])
    avg_rec = np.mean(avg_res_names[1])
    avg_f1 = np.mean(avg_res_names[2])
    print ("{}\tName-listing: Avg_micro_precision = {:0.3f}, Avg_micro_recall = {:0.3f}, Avg_micro_f1 = {:0.3f}".format(
                                                                      model_name, avg_prec, avg_rec, avg_f1))
    return 
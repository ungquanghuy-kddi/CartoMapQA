import numpy as np
import os
import json
from eval import RMSE, R2_score, \
                 Precision_name_listing, Recall_name_listing, F1_score_name_listing, \
                 Accuracy, Pre_recall_f1

TYPE = {
    "presence": ["presence"],
    "counting": ["counting"],
    "name_listing": ["name_listing"],
    "all": ["presence", "counting", "name_listing"],
}

def evaluate_stmf(model_name="gemini-2.5-pro-preview-03-25", 
                  eval_dir="./Responses/STMF/gemini-2.5-pro-preview-03-25", type="all"):
    res_type1 = [[], []] # true, pred
    res_type2 = [[], []] # true, pred
    res_type3 = [[], [], []] # prec, rec, f1
    for item in TYPE[type]:
        for fn in os.listdir(os.path.join(eval_dir, item)):
            with open(os.path.join(eval_dir, item, fn), 'r') as f:
                data = json.load(f)
            if item == "presence":
                res_type1[0].append(data["Correct answer"].lower())
                res_type1[1].append(data["Answer"].lower())
            
            if item == "counting":
                res_type2[0].append(data["Correct answer"])
                res_type2[1].append(data["Answer"])

            if item == "name_listing":
                true = data["Correct answer"]
                pred = data["Answer"]

                # unable to answer or result no names found
                if len(pred) == 1:
                    if "there are no" in pred[0].lower() or \
                        "there is no" in pred[0].lower() or \
                        "I am unable to" in pred[0] or \
                        "I'm not able" in pred[0] or \
                        pred[0] == "None":
                        pred = []

                # remove the last "." in pred
                if len(pred) > 0 and len(pred[-1]) > 0 and pred[-1][-1] == ".":
                    pred[-1] = pred[-1][:-1]

                # remove "<|eot|>" in pred
                if len(pred) > 0 and "<|eot|>" in pred[-1]:
                    pred[-1] = pred[-1].replace("<|eot|>", "")

                # remove "\"\"" in pred
                pred = [item for item in pred if item != "\"\""]

                prec = Precision_name_listing(true, pred)
                rec = Recall_name_listing(true, data["Response"])
                f1 = F1_score_name_listing(prec, rec)
                res_type3[0].append(prec)
                res_type3[1].append(rec)
                res_type3[2].append(f1)

    ### Eval presence question ###
    print ("============== STMF task =================")
    if "presence" in TYPE[type]:
        acc = Accuracy(res_type1[0], res_type1[1])
        pre, rec, f1 = Pre_recall_f1(res_type1[0], res_type1[1])
        print ("{}\tPresence: Accuracy = {:0.3f}, Precision = {:0.3f}, Recall = {:0.3f}, F1-score = {:0.3f}".format(model_name, acc, pre, rec, f1))

    ### Eval counting question ###
    if "counting" in TYPE[type]:
        rmse = RMSE(res_type2[0], res_type2[1])
        r2 = R2_score(res_type2[0], res_type2[1])
        print ("{}\tCounting: RMSE = {:0.3f}, R2 = {:0.3f}".format(model_name, rmse, r2))

    ### Eval name-listing question ###
    if "name_listing" in TYPE[type]:
        Avg_prec = np.mean(res_type3[0])
        Avg_rec = np.mean(res_type3[1])
        Avg_f1 = np.mean(res_type3[2])
        print ("{}\tName-listing: Avg_precision = {:0.3f}, Avg_recall = {:0.3f}, Avg_f1 = {:0.3f}".format(model_name, Avg_prec, Avg_rec, Avg_f1))

    return


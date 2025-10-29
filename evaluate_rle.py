import os
import json

from eval import RMSE, MAE, MAPE, R2_score

def evaluate_rle(model_name="o1-2024-12-17", 
                    eval_dir="./Responses/RLE/o1-2024-12-17"):    
    simple_true = [[], []] # meter, feet
    complex_true = [[], []] # meter, feet

    simple_pred = [[], []] # meter, feet
    complex_pred = [[], []] # meter, feet

    for i, fn in enumerate(os.listdir(eval_dir)):
        with open(os.path.join(eval_dir, fn), 'r') as f:
            data = json.load(f)

        if float(data["Result"]) == -1:
            continue

        if data["Difficulty"] == "Simple":
            if data["Measure"] == "meters":
                simple_true[0].append(float(data["Correct_answer"]))
                simple_pred[0].append(float(data["Result"]))
            elif data["Measure"] == "feet":
                simple_true[1].append(float(data["Correct_answer"]))
                simple_pred[1].append(float(data["Result"]))

        elif data["Difficulty"] == "Complex":
            if data["Measure"] == "meters":
                complex_true[0].append(float(data["Correct_answer"]))
                complex_pred[0].append(float(data["Result"]))
            elif data["Measure"] == "feet":
                complex_true[1].append(float(data["Correct_answer"]))
                complex_pred[1].append(float(data["Result"]))

        else:
            print ("Out of scope")

    print ("============== RLE task ===============")
    #### calculate MAE and RMSE ####
    # Simple, meter
    sim_m_rmse = round(RMSE(simple_true[0], simple_pred[0]), 2)
    sim_m_mae = round(MAE(simple_true[0], simple_pred[0]), 2)
    sim_m_mape = round(MAPE(simple_true[0], simple_pred[0]), 2)
    sim_m_r2 = round(R2_score(simple_true[0], simple_pred[0]), 2)

    print (model_name)
    print ("+ Simple & meter:")
    print (f"\tRMSE = {sim_m_rmse}, MAE = {sim_m_mae}, MAPE = {sim_m_mape}, R2 = {sim_m_r2}")

    # Simple, feet
    sim_f_rmse = round(RMSE(simple_true[1], simple_pred[1]), 2)
    sim_f_mae = round(MAE(simple_true[1], simple_pred[1]), 2)
    sim_f_mape = round(MAPE(simple_true[1], simple_pred[1]), 2)
    sim_f_r2 = round(R2_score(simple_true[1], simple_pred[1]), 2)


    print ("+ Simple & feet:")
    print (f"\tRMSE = {sim_f_rmse}, MAE = {sim_f_mae}, MAPE = {sim_f_mape}, R2 = {sim_f_r2}")

    # Complex, meter
    com_m_rmse = round(RMSE(complex_true[0], complex_pred[0]), 2)
    com_m_mae = round(MAE(complex_true[0], complex_pred[0]), 2)
    com_m_mape = round(MAPE(complex_true[0], complex_pred[0]), 2)
    com_m_r2 = round(R2_score(complex_true[0], complex_pred[0]), 2)

    print ("+ Complex & meter:")
    print (f"\tRMSE = {com_m_rmse}, MAE = {com_m_mae}, MAPE = {com_m_mape}, R2 = {com_m_r2}")

    # Complex, feet
    com_f_rmse = round(RMSE(complex_true[1], complex_pred[1]), 2)
    com_f_mae = round(MAE(complex_true[1], complex_pred[1]), 2)
    com_f_mape = round(MAPE(complex_true[1], complex_pred[1]), 2)
    com_f_r2 = round(R2_score(complex_true[1], complex_pred[1]), 2)

    print ("+ Complex, feet:")
    print (f"\tRMSE = {com_f_rmse}, MAE = {com_f_mae}, MAPE = {com_f_mape}, R2 = {com_f_r2}")

    return

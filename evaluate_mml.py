import os
import json


def evaluate_mml(model_name = "gemini-2.5-pro-preview-03-25", eval_dir="./Responses/MML/gemini-2.5-pro-preview-03-25"):
    """
    Evaluate the MML task results.
    This function reads the evaluation results from a specified directory,
    calculates the accuracy of the model's answers, and prints the results.
    """

    correct_count = 0
    correct_count_blue = 0
    correct_count_red = 0
    zoom_level_acc = {}
    for fn in os.listdir(eval_dir):
        with open(os.path.join(eval_dir, fn), 'r') as f:
            data = json.load(f)
            if data['Answer'] is not None:
                if ((data['Answer']['road_1'] == data['road_1'] and data['Answer']['road_2'] == data['road_2']) or 
                    (data['Answer']['road_1'] == data['road_2'] and data['Answer']['road_2'] == data['road_1'])):
                    correct_count += 1
                    if data['Marker_color'] == "blue":
                        correct_count_blue += 1
                    elif data['Marker_color'] == "red":
                        correct_count_red += 1
                    
                    if data['zoom_level'] not in zoom_level_acc:
                        zoom_level_acc[data['zoom_level']] = [1, 0]
                    else:
                        zoom_level_acc[data['zoom_level']][0] += 1
                
                if data['zoom_level'] not in zoom_level_acc:
                    zoom_level_acc[data['zoom_level']] = [0, 1]
                else:
                        zoom_level_acc[data['zoom_level']][1] += 1
    accuracy = correct_count / len(os.listdir(eval_dir))
    accuracy_blue = correct_count_blue / (len(os.listdir(eval_dir)) / 2)
    accuracy_red = correct_count_red / (len(os.listdir(eval_dir)) / 2)

    print ("============== MML task =================")
    print (f"Model: {model_name}")
    print ("Overall accuracy = {:0.3f} (correctly answer {} per total {} questions)".format(accuracy, correct_count, len(os.listdir(eval_dir))))
    print ("Accuracy_red = {:0.3f} (correctly answer {} per total {} questions)".format(accuracy_red, correct_count_red, int(len(os.listdir(eval_dir)) / 2)))
    print ("Accuracy_blue = {:0.3f} (correctly answer {} per total {} questions)".format(accuracy_blue, correct_count_blue, int(len(os.listdir(eval_dir)) / 2)))
    
    print ("*Zoom level accuracy:")
    for item in zoom_level_acc:
        print ("== Zoom level {}: {:0.3f}".format(item, zoom_level_acc[item][0] / zoom_level_acc[item][1]))
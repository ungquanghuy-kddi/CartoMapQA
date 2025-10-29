from eval import route_eval, check_valid_route
import os
import json


def evaluate_srn(model_name="o3-2025-04-16", eval_dir="./Responses/SRN/o3-2025-04-16",
                 road_conj_path = "./Dataset/CartoMapQA/MapNavigation/road_conjunction",
                 road_dict_path = "./Dataset/CartoMapQA/MapNavigation/road_dict"):
    zoom_level = [17, 18, 19]
    
    best_path_count = 0
    correct_step_overall = []
    connectivity_overall = []
    zoom_level_acc = {zlv: [0, 0] for zlv in zoom_level}
    correct_steps = {zlv: [] for zlv in zoom_level}
    connectivity = {zlv: [] for zlv in zoom_level}

    for fn in os.listdir(eval_dir):
        with open(os.path.join(eval_dir, fn), 'r') as f:
            data = json.load(f)
        response = data['Answer']
        gt = data['route_directions']
        gt = gt.replace("[", "").replace("]", "")
        groundtruth = [item.strip() for item in gt.split(",")]

        # load road conjunction
        with open(os.path.join(road_conj_path, fn), 'r') as f:
            conjunction = json.load(f)

        # load road dictionary
        with open(os.path.join(road_dict_path, fn), 'r') as f:
            road_dict = json.load(f)

        is_success, correct_step_count = route_eval(groundtruth, response)
        correct_steps[data['zoom_level']].append( (correct_step_count-1) / (len(groundtruth)-1)) # remove the start point
        correct_step_overall.append( (correct_step_count-1) / (len(groundtruth)-1)) # remove the start point
        
        if is_success:
            best_path_count += 1
            zoom_level_acc[data['zoom_level']][0] += 1
            connectivity[data['zoom_level']].append(True)
            connectivity_overall.append(True)
        else:
            is_valid = check_valid_route(origin=data["origin"],
                                        dest=data["destination"],
                                        path=data['Answer'],
                                        road_dict=road_dict,
                                        conj_dict=conjunction)
            connectivity[data['zoom_level']].append(is_valid)
            connectivity_overall.append(is_valid)


        zoom_level_acc[data['zoom_level']][1] += 1

    print ("============== SRN task =================")
    print (f"Model: {model_name}")
    for item in zoom_level_acc:
        step_accuracy = sum(correct_steps[item]) / len(correct_steps[item]) 
        connectivity_rate = sum(connectivity[item]) / len(connectivity[item]) 
        print ("Zoom level {}: shortest_path_success_rate = {:0.3f}, avg_step_accuracy = {:0.3f}, connectivity_rate = {:0.3f}".format(item,
                                    zoom_level_acc[item][0] / zoom_level_acc[item][1],
                                    step_accuracy, connectivity_rate))

    print ("==== Overall ====")
    print ("shortest_path_success_rate: {:0.3f}".format(best_path_count / len(os.listdir(eval_dir))))
    print ("avg_step_accuracy: {:0.3f}".format(sum(correct_step_overall) / len(correct_step_overall)))
    print ("avg_connectivity_rate: {:0.3f}".format(sum(connectivity_overall) / len(connectivity_overall)))

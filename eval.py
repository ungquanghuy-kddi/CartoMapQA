from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import r2_score
import math

def R2_score(true, pred):
    r2 = r2_score(true, pred)
    return r2

def MAPE(true, pred):
    mape = mean_absolute_percentage_error(true, pred)
    return mape

def RMSE(true, pred):
    rmse = sqrt(mean_squared_error(true, pred))
    return rmse

def MAE(true, pred):
    mae = mean_absolute_error(true, pred)
    return mae

def Accuracy(true, pred):
    correct_count = 0
    for t, p in zip(true, pred):
        if t == p:
            correct_count += 1
    Acc = correct_count / len(true)
    return Acc

def Pre_recall_f1(true, pred):
    true_num = []
    for item in true:
        if item == "yes":
            true_num.append(1)
        if item == "no":
            true_num.append(0)

    pred_num = []
    for item in pred:
        if item == "yes":
            pred_num.append(1)
        if item == "no":
            pred_num.append(0)

    precision = precision_score(true_num, pred_num, average='macro')
    recall = recall_score(true_num, pred_num, average='macro')
    f1 = f1_score(true_num, pred_num, average='macro')

    return precision, recall, f1

def Precision_name_listing(true, pred):
    true = [item.strip() for item in true]
    pred = [item.strip() for item in pred]

    merge_list = list(set(true + pred))
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(merge_list)
    true_encoded = label_encoder.transform(true)
    pred_encoded = label_encoder.transform(pred)

    TP = len(set(true_encoded).intersection(set(pred_encoded)))

    if len(pred) != 0:
        precision = TP * 1.0 / len(pred)
    else:
        precision = 1 if len(true) == 0 else 0

    return precision

def Recall_name_listing(true, response):
    TP = 0
    for item in true:
        if item in response:
            TP += 1
    
    if len(true) != 0:
        recall = TP * 1.0 / len(true)
    else:
        recall = 1 if len(true) == 0 else 0

    return recall

def F1_score_name_listing(prec, recall):
    if prec + recall == 0:
        return 0
    else:
        return 2 * (prec * recall) * 1.0 / (prec + recall)
    
def normalize_route(route):
    """
    Normalize the route by removing "continue straight" on a road.
    """
    idx_to_remove = []
    for idx, action in enumerate(route):
        if action == "continue straight" \
            and idx+1 < len(route) \
            and idx-1 >= 0 \
            and route[idx+1] == route[idx-1]:
            idx_to_remove.append(idx)
            idx_to_remove.append(idx+1)
    
    for idx in sorted(idx_to_remove, reverse=True):
        route.pop(idx)
    return route

def route_eval(true, answer):
    
    correct_flag = True
    correct_step_count = 0
    for t_step, a_step in zip(true, answer):
        if t_step != a_step:
            correct_flag = False
            break
        else:
            correct_step_count += 1
    if correct_flag:
        return 1, correct_step_count

    return 0, correct_step_count

def is_turning_error(true, answer):
    if answer[0] != "blue" or answer[-1] != "red":
        return False
    
    if len(true) != len(answer):
        return False

    # check road names
    for idx, (t, a) in enumerate(zip(true, answer)):
        if idx%2 == 0 and t != a:
            return False

    # road names are correct
    # check the direction
    for idx, (t, a) in enumerate(zip(true, answer)):
        if idx%2 == 1 and t != a:
            return [t, a]
    
    return None

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two points.
    """
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    d_lon = lon2 - lon1
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(d_lon))
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def get_turn_direction_(bearing1, bearing2):
    """
    Determine the turn direction based on the change in bearing.
    """
    diff = (bearing2 - bearing1 + 360) % 360
    if diff < 30 or diff > 330:
        return "continue straight"
    elif diff < 150:
        return "turn right"
    elif diff > 210:
        return "turn left"
    else:
        return "make a U-turn and continue straight"

def get_turn_direction(bearing1, bearing2):
    """
    Determine the turn direction based on the change in bearing.
    """
    diff = (bearing2 - bearing1 + 360) % 360
    if diff < 40 or diff > 320:
        return "continue straight"
    elif diff < 140:
        return "turn right"
    elif diff > 220:
        return "turn left"
    else:
        return "make a U-turn and continue straight"

def generate_route_directions(route, road_names):
    directions = []
    current_bearing = 0  # Assuming the user is facing north initially

    for i in range(len(route) - 1):
        node1 = route[i]
        node2 = route[i + 1]

        lat1, lon1 = float(node1[1]), float(node1[2])
        lat2, lon2 = float(node2[1]), float(node2[2])

        next_bearing = calculate_bearing(lat1, lon1, lat2, lon2)
        turn_direction = get_turn_direction(current_bearing, next_bearing)
        road_name = road_names[i]

        directions.extend([turn_direction, road_name])
        current_bearing = next_bearing

    output = ["blue"]
    for idx, direction in enumerate(directions):
        output.append(direction)
    output.append("red")

    return output

def check_valid_route(origin, dest, path, conj_dict, road_dict, verbose=False):
    if path[0] != "blue" or path[-1] != "red":
        return False # path should start with blue and end with red

    # check roads of origin and destination
    road_origin = path[2]
    road_destination = path[-2]

    ## roads are not in the road_dict
    if road_origin not in road_dict:
        if verbose:
            print ("road containing origin is not in the road_dict")
        return False # road is not in the road_dict
    
    if road_destination not in road_dict:
        if verbose:
            print ("road containing destination is not in the road_dict")
        return False # road is not in the road_dict

    origin_check = False
    for link in road_dict[road_origin]:
        for node in road_dict[road_origin][link]:
            if str(origin[0]) == str(node[0]):
                origin_check = True
                break

    dest_check = False
    for link in road_dict[road_destination]:
        for node in road_dict[road_destination][link]:
            if str(dest[0]) == str(node[0]):
                dest_check = True
                break

    if not origin_check:
        if verbose:
            print ("origin is not on the road")
        return False # origin is not on the road

    if not dest_check:
        if verbose:
            print ("destination is not on the road")
        return False # destination is not on the road

    ## check the path conectivity
    list_of_action = ["left", "right", "straight", "u-turn"]
    street_names = []
    for item in path:
        is_street = True
        for action in list_of_action:
            if action in item:
                is_street = False
                break
        # if is_street and item != "A" and item != "B":
        if is_street and item != "blue" and item != "red":
            street_names.append(item)
    
    if len(street_names) > 1: 
        list_juction_node = []
        ## check all road names in the path
        for name in street_names:
            if name not in road_dict:
                if verbose:
                    print (f"incorrect road names: {name}")
                return False

        ## check connectivity between roads
        for idx in range(len(street_names) - 1, 0, -1):
            if street_names[idx] == street_names[idx-1]:
                street_names.pop(idx)

        for i in range(len(street_names) - 1):
            if street_names[i] not in conj_dict[street_names[i+1]]:
                if verbose:
                    print (f"incorrect connectivity between {street_names[i]} and {street_names[i+1]}")
                return False # the path is not connected
            else:
                list_juction_node.append(conj_dict[street_names[i+1]][street_names[i]][0])
        
        # print (street_names)
        list_juction_node.insert(0, origin)
        list_juction_node.append(dest)
        ## check direction change
        # generate the direction of the path
        correct_path_with_direction = generate_route_directions(list_juction_node, street_names)
 
        for item1, item2 in zip(path, correct_path_with_direction):
            if item1 != item2:
                if verbose:
                    print (f"incorrect direction at {item1}")
                return False

    return True


import json
POI_ALTER_NAME = {
    "0": ["convenience store", "convenience_store"],
    "1": ["supermarket"],
    "2": ["drug store/pharmacy", "drug_store/pharmacy"],
    "3": ["general clinic", "general_clinic"],
    "4": ["hospital"],
    "5": ["restaurant"],
    "6": ["fast food store", "fast_food_store"],
    "7": ["coffee shop", "coffee_shop"],
    "8": ["atm or cash point", "atm_or_cash_point"],
    "9": ["bank"],
    "10": ["playground"],
    "11": ["park area", "park_area"],
    "12": ["fitness center", "fitness_center"],
    "13": ["clothes shop", "clothes_shop"],
    "14": ["beauty shop", "beauty_shop"],
    "15": ["water stream", "water_stream"],
    "16": ["river"],
    "17": ["bus stop", "bus_stop"],
    "18": ["gas station", "gas_station"],
    "19": ["primary road", "primary_road"],
    "20": ["car parking lot", "car_parking_lot"],
    "21": ["public toilet", "public_toilet"],
}

POI_LIST = {
    "0": "convenience store",
    "1": "supermarket",
    "2": "drug store/pharmacy",
    "3": "general clinic",
    "4": "hospital",
    "5": "restaurant",
    "6": "fast food store",
    "7": "coffee shop",
    "8": "ATM or cash point",
    "9": "bank",
    "10": "playground",
    "11": "park area",
    "12": "fitness center",
    "13": "clothes shop",
    "14": "beauty shop",
    "15": "water stream",
    "16": "river",
    "17": "bus stop",
    "18": "gas station",
    "19": "primary road",
    "20": "car parking lot",
    "21": "public toilet"
}

def prompt_mtmf():
    exp_json = {
  "Convenience store": {"count": 2, "names": ["name_cs_1", "name_cs_2"]},
  "Hospital": {"count": 0, "names": []},
  "Playground": {"count": 3, "names": ["", "", "name_pg_3"]},
  "temp": "temp",
  "Public toilet": {"count": 0, "names": []}
}
    json_string = json.dumps(exp_json)
    json_string = json_string.replace('"temp": "temp"', "...")
    json_string = "```json" + json_string + "```"
    
    prompt = f"""You are provided with a cartographic map sourced from OpenStreetMap, along with the definitions of point-of-interest (POI) types used in OpenStreetMap.\n\
  Your task is to:\n\
  Identify visual elements in the map that correspond to the following list of POI types.\n\
  Count how many of such elements appear for each POI type.\n\
  List the associated names for each identified POI element, if available. If a POI element has no name, represent it with an empty string ("").\n\
  Return your results in a single JSON object where each POI type is a key, and its value is an object with the following two keys:\n\
  "count": the total number of corresponding visual elements found.\n\
  "names": a list of names associated with the found elements (use exact names as shown on the map; use "" for unnamed elements).\n\
  If no visual elements are found for a given POI, set "count" to 0 and "names" to an empty list [].\n\

  Example output:\n\
  {json_string}\n\
  Perform this task for the following POI types:\n\
  Convenience store\n\
  Supermarket\n\
  Drug store/pharmacy\n\
  General clinic\n\
  Hospital\n\
  Restaurant\n\
  Fast food store\n\
  Coffee shop\n\
  ATM or cash point\n\
  Bank\n\
  Playground\n\
  Park area\n\
  Fitness center\n\
  Clothes shop\n\
  Beauty shop\n\
  Water stream\n\
  River\n\
  Bus stop\n\
  Gas station\n\
  Primary road\n\
  Car parking lot\n\
  Public toilet\n\
  Strictly follow this format. Do not include any explanation, description, code, or additional text—only output the JSON object as shown."""
    return prompt

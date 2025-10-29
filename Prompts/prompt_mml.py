import json

def prompt_mml(marker_color):
    """
    Prompt for map marker localization task.
    """
    exp_json = {'road_1': "First Street", 'road_2': "Main Avenue"}
    json_string = json.dumps(exp_json)
    prompt = f"""You are provided with a cartographic map sourced from OpenStreetMap, where a {marker_color} marker indicates the intersection of two roads.\n\
Your task is to identify the names of the two roads intersecting at that point. These may include highways, avenues, secondary roads, or tertiary roads.\n\
Use the exact names as they appear on the map, do not use information that are not in the map.\n\
Return your result as a single JSON object using the following format:\n\
```json{json_string}```\n\
Strictly follow this format. Do not include any explanation, description, code, or additional text—only output the JSON object."""
    return prompt
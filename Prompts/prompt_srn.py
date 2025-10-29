
def prompt_srn():
    prompt = f"""You are provided with a cartographic map sourced from OpenStreetMap. Two colored map markers are shown: the blue marker indicates the starting location, and the red one marks the destination.\n\
Assume the user is located at the blue marker, seated in a vehicle, and initially facing toward the top of the map. Your task is to determine the shortest drivable route from the blue marker to the red marker, and describe all required driving actions to reach the destination.\n\
Output your answer using the following strict format: Answer: [blue, <action_1>, road_1, <action_2>, road_2, ..., <action_N>, road_N, red]\n\
Each <action> must be one of the following:\n\
- "make a U-turn and continue straight"\n\
- "continue straight"\n\
- "turn left"\n\
- "turn right"\n\
Use the exact road names as shown on the map.\n\
For example: If the route involves a u-turn at the start, followed by driving on "Main Street", a left turn onto "2nd Avenue", and a right turn onto "Elm Street" to reach the destination, the response should be:\
Answer: [blue, make a U-turn and continue straight, Main Street, turn left, 2nd Avenue, turn right, Elm Street, red]\n\
Strictly follow this format. Do not include any explanation, justification, or additional text—only return the final answer list with the format as shown above."""
    return prompt
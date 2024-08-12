import re
import json 

def extract_json_from_string(string_data):
    """
    Extracts JSON data from a given string.

    Parameters:
    string_data (str): The input string containing JSON data.

    Returns:
    dict or list: The extracted JSON data as a Python object (dict or list).
    """
    # Use regular expressions to find the JSON data within the string
    json_data_match = re.search(r'```json(.*?)```', string_data, re.DOTALL)

    # Check if JSON data was found
    if json_data_match:
        json_data = json_data_match.group(1).strip()
        # Convert the JSON string to a Python object
        json_object = json.loads(json_data)
        return json_object
    else:
        raise ValueError("No JSON data found in the string.")
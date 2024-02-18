import os.path
import sys
import argparse
import json


def parse_string_to_numbers(input_string):
    # Split the string by commas to process each part
    parts = input_string.split(',')
    numbers = []
    for part in parts:
        # Check if part contains a dash, indicating a range
        if '-' in part:
            start, end = map(int, part.split('-'))
            # Generate all numbers in the range and add them to the list
            numbers.extend(range(start, end + 1))
        else:
            # Part is a single number, convert to int and add to the list
            numbers.append(int(part))
    return numbers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="set the node properties of a script file", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("script_file_path", help="script file path", type=str)
    parser.add_argument("tick", help="tick of setting the properties", type=int)
    parser.add_argument("node_list", help="the node list to be set properties", type=str)
    parser.add_argument("property_name", help="property name", type=str)
    parser.add_argument("property_value", help="property value", type=str)

    args = parser.parse_args()
    config = vars(args)
    if len(sys.argv) < 5:
        print("use: python ./script_set_nodes_properties.py {node_list} {property_name} {property_value}")
        exit(1)

    script_file_path = str(config['script_file_path'])
    tick = int(config['tick'])
    target_nodes_str = str(config['node_list'])
    property_name_str = str(config['property_name'])
    property_value_str = str(config['property_value'])

    if not os.path.exists(script_file_path):
        print(f"script file does not exist: {script_file_path}")
        exit(1)
    with open(script_file_path, mode='r') as script_file:
        script_file_json = json.loads(script_file.read())
    target_nodes = parse_string_to_numbers(target_nodes_str)
    value = property_value_str
    if value.lower() == "true":
        value = True
    elif value.lower() == "false":
        value = False
    elif value.isdigit():
        value = int(value)
    elif value.isnumeric():
        value = float(value)

    # change property
    changed_nodes = {}
    for node in target_nodes:
        changed_nodes[str(node)] = {
            property_name_str: value
        }
    script_file_json.append({
        "tick": tick,
        "script": changed_nodes,
    })

    # backup
    os.rename(script_file_path, f"{script_file_path}.bak")

    # write new file
    new_script_file_content = json.dumps(script_file_json, indent=4)
    with open(script_file_path, mode='w') as new_script_file:
        new_script_file.write(new_script_file_content)



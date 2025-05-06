def get_measured_angle():
    import json
    with open("../data.json", "r") as json_file:
        loaded_data = json.load(json_file)
    return loaded_data['measured_angle']


def get_target_angle():
    import json
    with open("../data.json", "r") as json_file:
        loaded_data = json.load(json_file)
    return loaded_data['target_angle']
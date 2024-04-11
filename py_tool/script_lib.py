import json
import networkx as nx


def save_script_to_file(script_target, file_path: str):
    f = open(file_path, "w")
    f.write(json.dumps(script_target, indent=4))
    f.close()


def __convert_arg_to_list__(args):
    if isinstance(args, int):
        return [args]
    elif isinstance(args, list):
        return args
    else:
        raise Exception("arg has to be int or [int]")


def set_node_status(script_target, tick: int, nodes, enabled: bool):
    nodes = __convert_arg_to_list__(nodes)
    node_config = {
        "enable": enabled
    }
    changed_nodes = {}
    for node in nodes:
        changed_nodes[str(node)] = node_config
    script_target.append({
        "tick": tick,
        "script": changed_nodes,
    })


# normal,
# observer,       //node that only performs averaging, will not send models
# no_training,    //nodes will not perform training, but will send models
# pontificator,   //nodes will always propagate its initial model
# malicious_model_poisoning_random_model,             //always send random models
# malicious_model_poisoning_random_model_by_turn,     //send random models by turns(one good model + one malicious model)
# malicious_model_poisoning_random_model_biased_0_1,  //subtract 0-0.1 from the model weights
# malicious_duplication_attack,                       //duplicate other's model and send it to others
# malicious_data_poisoning_shuffle_label,             //shuffle the label of training dataset and send it as training dataset to nodes
# malicious_data_poisoning_shuffle_label_biased_1,    //add 1 to the label of training dataset and send it as training dataset to nodes
# malicious_data_poisoning_random_data,               //use uniformly random training dataset to train the model
# normal_label_0_4,   //normal nodes but only receive training dataset whose labels are from 0 to 4
# normal_label_5_9,   //normal nodes but only receive training dataset whose labels are from 5 to 9
def set_node_type(script_target, tick: int, nodes, node_type: str):
    nodes = __convert_arg_to_list__(nodes)
    node_config = {
        "node_type": node_type
    }
    changed_nodes = {}
    for node in nodes:
        changed_nodes[str(node)] = node_config
    script_target.append({
        "tick": tick,
        "script": changed_nodes,
    })


def set_node_buffer_size(script_target, tick: int, nodes, buffer_size):
    nodes = __convert_arg_to_list__(nodes)
    buffer_size = __convert_arg_to_list__(buffer_size)
    changed_nodes = {}
    assert(len(buffer_size) == len(nodes))
    for index, node in enumerate(nodes):
        if len(buffer_size) == 1:
            node_config = {
                "buffer_size": buffer_size[0]
            }
        else:
            node_config = {
                "buffer_size": buffer_size[index]
            }
        changed_nodes[str(node)] = node_config
    script_target.append({
        "tick": tick,
        "script": changed_nodes,
    })


def set_node_training_interval_tick(script_target, tick: int, nodes, training_interval_tick):
    nodes = __convert_arg_to_list__(nodes)
    changed_nodes = {}
    node_config = {
        "training_interval_tick": training_interval_tick
    }
    for node in nodes:
        changed_nodes[str(node)] = node_config
    script_target.append({
        "tick": tick,
        "script": changed_nodes,
    })


def set_node_model_override_model_from(script_target, tick: int, nodes, src_nodes):
    nodes = __convert_arg_to_list__(nodes)
    src_nodes = __convert_arg_to_list__(src_nodes)
    changed_nodes = {}
    assert(len(src_nodes) == len(nodes) or len(src_nodes) == 1)
    for index, node in enumerate(nodes):
        if len(src_nodes) == 1:
            node_config = {
                "override_model_from": str(src_nodes[0])
            }
        else:
            node_config = {
                "override_model_from": str(src_nodes[index])
            }
        changed_nodes[str(node)] = node_config
    script_target.append({
        "tick": tick,
        "script": changed_nodes,
    })


def set_node_weights(script_target, tick:int, value: float, nodes):
    nodes = __convert_arg_to_list__(nodes)
    changed_nodes = {}
    for index, node in enumerate(nodes):
        node_config = {
            "set_weights": value
        }
        changed_nodes[str(node)] = node_config
    script_target.append({
        "tick": tick,
        "script": changed_nodes,
    })


def scale_node_weights(script_target, tick:int, value: float, nodes):
    nodes = __convert_arg_to_list__(nodes)
    changed_nodes = {}
    for index, node in enumerate(nodes):
        node_config = {
            "scale_weights": value
        }
        changed_nodes[str(node)] = node_config
    script_target.append({
        "tick": tick,
        "script": changed_nodes,
    })


def find_edges_among_two_blocks(G: nx.Graph, block_a, block_b):
    edges_between_a_and_b = []
    for node in G.nodes:
        if node not in block_a:
            continue
        for peer in list(G.neighbors(node)):
            if peer in block_b:
                edges_between_a_and_b.append({node, peer})
    return edges_between_a_and_b



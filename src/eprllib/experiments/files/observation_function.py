"""
Observation Functions
======================


"""

def task_policy_map_fn(agent_id, episode, worker, **kwargs):
    assert type(agent_id) == str , "Agent ID must not be None."
    if agent_id == "HVAC":
        return f"{agent_id}_policy"
    # elif agent_id ends with "Window" -> "NV_policy"
    elif agent_id.endswith("Windows"):
        return "NV_policy"
    elif agent_id.endswith("Shades"):
        return "WSC_policy"
"""
This script is used to train AV agents using the baseline methods in a traffic simulation environment.
Baseline methods can be found in the baseline_models/ directory.
"""

import os
import sys


os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import ast
import json
import logging
import random

import numpy as np
import pandas as pd

from routerl import Keychain as kc
from routerl import TrafficEnvironment
from routerl.environment import HumanAgent
from tqdm import tqdm

from utils import clear_SUMO_files
from baseline_models import get_baseline
from baseline_models.tabql import TabularQLearning

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    ALGORITHM = "baseline"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    baseline_model = "tabql"
    #assert baseline_model in kc.HUMAN_MODELS, f"Model {baseline_model} not in {kc.HUMAN_MODELS}"
    print("### STARTING EXPERIMENT ###")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")
    print(f"Baseline model: {baseline_model}")

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    random.seed(env_seed)
    np.random.seed(env_seed)
        
    # Parameter setting
    params = dict()
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"], alg_params, env_params, task_params

    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value

    
    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"

    # Read origin-destinations
    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data['origins']
    destinations = data['destinations']

    
    # Copy agents.csv from custom_network_folder to records_folder
    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(new_agents_csv_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps
            
    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["env_config"] = env_config
    dump_config["task_config"] = task_config
    dump_config["alg_config"] = alg_config
    dump_config["baseline_model"] = baseline_model
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    # Initiate the traffic environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = True,
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": num_machines, 
            "human_parameters" : {
                "model" : human_model
            },
            "machine_parameters" :{
                "behavior" : av_behavior,
            }
        },
        environment_parameters = {
            "save_every" : save_every,
        },
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo",
        }, 
        plotter_parameters = {
            "phases" : phases,
            "phase_names" : phase_names,
            "smooth_by" : smooth_by,
            "plot_choices" : plot_choices,
            "records_folder" : records_folder,
            "plots_folder" : plots_folder
        },
        path_generation_parameters = {
            "origins" : origins,
            "destinations" : destinations,
            "number_of_paths" : number_of_paths,
            "beta" : path_gen_beta,
            "num_samples" : num_samples,
            "visualize_paths" : False
        } 
    )

    print(f"""
    Agents in the traffic:
    • Total agents           : {len(env.all_agents)}
    • Human agents           : {len(env.human_agents)}
    • AV agents              : {len(env.machine_agents)}
    """)

    
    env.start()
    res = env.reset()

    # Human learning
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()

    #  Mutation
    pre_mutation_agents = env.all_agents.copy()
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile = -1)

    print(f"""
    Agents in the traffic:
    • Total agents           : {len(env.all_agents)}
    • Human agents           : {len(env.human_agents)}
    • AV agents              : {len(env.machine_agents)}
    """)

    # Replace AV models with baseline models
    machines = env.machine_agents.copy()
    mutated_humans = dict()

    for machine in machines:
        for human in pre_mutation_agents:
            if human.id == machine.id:
                mutated_humans[str(machine.id)] = human
                break
            
    human_learning_params = env.agent_params[kc.HUMAN_PARAMETERS]
    human_learning_params["model"] = baseline_model
    free_flows = env.get_free_flow_times()
    for h_id, human in mutated_humans.items():
        initial_knowledge = free_flows[(human.origin, human.destination)]
        initial_knowledge = [-1 * item for item in initial_knowledge]
        mutated_humans[h_id].model = TabularQLearning(number_routes=4)
       
    # Training
    pbar.set_description("AV learning")

    for episode in range(training_eps):
        env.reset()
        # Store previous step info for each agent
        prev_states = {}
        prev_actions = {}

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            # Convert observation to discrete state (you'll implement this)
            current_state = observation[1:] # Assuming this will be discrete

            done = termination or truncation
            if agent in prev_states:
                mutated_humans[agent].model.learn(
                    state=prev_states[agent],
                    action=prev_actions[agent], 
                    reward=reward,
                    next_state=current_state if not done else None,
                    done=done
                )

            if done:
                action = None
                # Clear stored info for this agent
                if agent in prev_states:
                    del prev_states[agent]
                    del prev_actions[agent]
            else:
                action = mutated_humans[agent].act(current_state)  # Pass state to act method
                # Store current step info for next iteration
                prev_states[agent] = current_state
                prev_actions[agent] = action
            env.step(action)
        pbar.update()
    
    # Testing
    pbar.set_description("Testing")
    for episode in range(test_eps):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action = mutated_humans[agent].act(observation)
            env.step(action)
        pbar.update()

    pbar.close()
    os.makedirs(plots_folder, exist_ok=True)
    env.plot_results()

    env.stop_simulation()

    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
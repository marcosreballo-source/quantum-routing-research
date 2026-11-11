import sys, os
sys.path.append(os.path.abspath('qamoo/notebooks/results'))
from utils import *
import numpy as np

configs = [
    # TSP 17->37, original
    {'prob': 'TSP', 'target': 37, 'origin': 17, 'p': 3, 'suffix': '200_samples_5000_shots_ibm_real'},
    # TSP 17->37, updated
    {'prob': 'TSP', 'target': 37, 'origin': 17, 'p': 3, 'suffix': '750_samples_5000_shots_ibm_real'},
    # VRP 27->43, original
    {'prob': 'VRP', 'target': 43, 'origin': 27, 'p': 2, 'suffix': '200_samples_5000_shots_ibm_real'},
    # VRP 27->43, updated
    {'prob': 'VRP', 'target': 43, 'origin': 27, 'p': 6, 'suffix': '750_samples_5000_shots_ibm_real'},
]

for cfg in configs:
    print(f"--- {cfg['prob']} {cfg['origin']}->{cfg['target']}, p={cfg['p']}, suffix={cfg['suffix']} ---")
    problem_folder = f"data/problems/{cfg['target']}q/problem_set_{cfg['target']}q_10s_3o_0/"
    
    try:
        all_data = load_progress_data(p_list=[cfg['p']], problem_folder=problem_folder, origin_num_qubits=cfg['origin'], suffix=cfg['suffix'])
        max_hv = all_data['max_hv'][cfg['p']]
        print(f"Max HV: {max_hv}")
    except Exception as e:
        print(f"Error loading max HV: {e}")
        
    try:
        results_folder = problem_folder + f"results/qaoa_MIPS_{cfg['origin']}q_p_{cfg['p']}_{cfg['suffix']}/"
        try:
            samples = np.load(results_folder + "non_dominated_samples_result.npy")
        except FileNotFoundError:
            all_samples = np.load(results_folder + "samples.npy")
            positions = np.load(results_folder + "non_dominated_positions.npy")
            samples = all_samples[positions]
            
        print(f"Number of ND solutions: {len(samples)}")
    except Exception as e:
        print(f"Error loading ND solutions: {e}")

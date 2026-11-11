import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('qamoo/notebooks/results'))
from utils import *
import pickle

os.makedirs("escrita_iop/figures", exist_ok=True)

# Custom function to plot single HV progress to avoid the 1x2 subplots from utils.py
def custom_plot_hv(data_hw, data_sim, filename, title):
    plt.figure(figsize=(6, 4))
    max_x = 0
    
    for p, x, y in data_hw:
        plt.step(x, y, label=f"Hardware (p={p})", where="post")
        if len(x) >= 2: max_x = max(max_x, x[-2])
        else: max_x = max(max_x, x[-1])
        
    for p, x, y in data_sim:
        plt.step(x, y, label=f"Simulation (p={p})", where="post", linestyle="--")
        if len(x) >= 2: max_x = max(max_x, x[-2])
        else: max_x = max(max_x, x[-1])
    
    plt.xlim(-1000, max_x * 1.05 + 1000)
    plt.grid(True)
    plt.xlabel("Evaluated Weights")
    plt.ylabel("Hypervolume")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"escrita_iop/figures/{filename}.pdf")
    plt.close()
    print(f"Saved {filename}.pdf")

# Patch plt.show for the solutions plot
original_show = plt.show
current_filename = None

def my_show():
    global current_filename
    if current_filename:
        plt.savefig(f"escrita_iop/figures/{current_filename}.pdf")
        print(f"Saved {current_filename}.pdf")
    plt.close()

plt.show = my_show

print("Extracting TSP HW plots...")
cfg = {'prob': 'TSP', 'target': 37, 'origin': 17, 'p': 3, 'suffix': '750_samples_5000_shots_ibm_real'}
problem_folder = f"qamoo/data/problems/{cfg['target']}q/problem_set_{cfg['target']}q_10s_3o_0/"
all_data_tsp = load_progress_data(p_list=[cfg['p']], problem_folder=problem_folder, origin_num_qubits=cfg['origin'], suffix=cfg['suffix'])
all_data_tsp_sim = load_progress_data(p_list=[cfg['p']], problem_folder=problem_folder, origin_num_qubits=cfg['origin'], suffix='5000_samples_5000_shots_20')

# Generate single HV plot
custom_plot_hv(all_data_tsp, all_data_tsp_sim, 'tsp_37q_hw_hv_evolution', 'TSP-6 Hardware HV Evolution')

# Generate solutions plot
current_filename = 'tsp_37q_hw_solutions'
with open(problem_folder + "problem_info.pkl", "rb") as f:
    info = pickle.load(f)
with open(problem_folder + "graph_instances.pkl", "rb") as f:
    instances = pickle.load(f)
results_folder = problem_folder + f"results/qaoa_MIPS_{cfg['origin']}q_p_{cfg['p']}_{cfg['suffix']}/"
try:
    samples = np.load(results_folder + "non_dominated_samples_result.npy")
except FileNotFoundError:
    all_samples = np.load(results_folder + "samples.npy")
    positions = np.load(results_folder + "non_dominated_positions.npy")
    samples = all_samples[positions]

if 'filter_duplicate_solutions' in globals():
    unique_solutions = filter_duplicate_solutions(samples, info)
else:
    unique_solutions = canonicalize_and_deduplicate(samples, info)

if 'plot_solutions' in globals():
    plot_solutions(unique_solutions, info, instances)
else:
    plot_maxcut_solutions(unique_solutions, info, instances)


print("Extracting VRP HW plots...")
cfg = {'prob': 'VRP', 'target': 43, 'origin': 27, 'p': 6, 'suffix': '750_samples_5000_shots_ibm_real'}
problem_folder = f"qamoo/data/problems/{cfg['target']}q/problem_set_{cfg['target']}q_10s_3o_0/"
all_data_vrp = load_progress_data(p_list=[cfg['p']], problem_folder=problem_folder, origin_num_qubits=cfg['origin'], suffix=cfg['suffix'])
all_data_vrp_sim = load_progress_data(p_list=[cfg['p']], problem_folder=problem_folder, origin_num_qubits=cfg['origin'], suffix='5000_samples_5000_shots_20')

# Generate single HV plot
custom_plot_hv(all_data_vrp, all_data_vrp_sim, 'vrp_43q_hw_hv_evolution', 'VRP-5-1 Hardware HV Evolution')

# Generate solutions plot
current_filename = 'vrp_43q_hw_solutions'
with open(problem_folder + "problem_info.pkl", "rb") as f:
    info = pickle.load(f)
with open(problem_folder + "graph_instances.pkl", "rb") as f:
    instances = pickle.load(f)
results_folder = problem_folder + f"results/qaoa_MIPS_{cfg['origin']}q_p_{cfg['p']}_{cfg['suffix']}/"
try:
    samples = np.load(results_folder + "non_dominated_samples_result.npy")
except FileNotFoundError:
    all_samples = np.load(results_folder + "samples.npy")
    positions = np.load(results_folder + "non_dominated_positions.npy")
    samples = all_samples[positions]
unique_solutions = canonicalize_and_deduplicate(samples, info)
plot_maxcut_solutions(unique_solutions, info, instances)

print("Done.")

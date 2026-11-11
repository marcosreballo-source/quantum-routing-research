import sys
import os

# Ensure qamoo is in path
sys.path.append(os.path.abspath('.'))

from qamoo.utils.utils import compute_hypervolume_progress

def compute_hw(target_num, problem_id, origin_num, p, samples):
    print(f"======================================")
    print(f"Computing Hypervolume for {target_num}q hardware results (p={p})")
    print(f"======================================")
    
    problem_folder = f"qamoo/data/problems/{target_num}q/problem_set_{target_num}q_10s_3o_{problem_id}/"
    results_folder = problem_folder + f"results/qaoa_MIPS_{origin_num}q_p_{p}_{samples}_samples_5000_shots_ibm_real/"
    
    if not os.path.exists(results_folder):
        print(f"Error: Folder {results_folder} does not exist.")
        return
        
    if not os.path.exists(results_folder + "samples.npy"):
        print(f"Error: samples.npy not found in {results_folder}.")
        return

    # Calculate steps incrementally for the progress JSON
    # Typically we might want steps every 50 samples, or just use the number of samples directly.
    # We can do steps = [0, 50, 100, 150, 200]
    steps = list(range(0, samples + 1, 50))
    if steps[-1] != samples:
        steps.append(samples)
        
    print(f"Steps for evaluation: {steps}")
    
    compute_hypervolume_progress(problem_folder, results_folder, steps)
    
    # compress.py equivalent
    import json
    import pandas as pd

    with open(results_folder + "progress.json") as f:
        progress = json.load(f)

    x = sorted(map(int, progress.keys()))
    y = [progress[str(i)] for i in x]

    df = pd.DataFrame({"x": x, "y": y})
    mask = (df["y"].shift() != df["y"]) | (df["y"].shift(-1) != df["y"])
    df = df[mask]
    
    x_comp, y_comp = df["x"].values, df["y"].values
    
    compressed_progress = {str(xi * 5000): yi for xi, yi in zip(x_comp, y_comp)}

    with open(results_folder + "progress_compressed.json", "w") as f:
        json.dump(compressed_progress, f, indent=2)
        
    print(f"Successfully computed and compressed hypervolume progress for {target_num}q (p={p}).\n")


if __name__ == "__main__":
    # 37q TSP
    compute_hw(37, 0, 17, 3, 750)
    
    # 43q VRP
    compute_hw(43, 0, 27, 6, 750)
    
    print("All done! You can now run the notebooks.")

import os


run_file_paths = [
    "splited_Pneumonia_all_true_false_split_1_(300, 300).py",
    "splited_Pneumonia_all_true_false_split_1_0_(300, 300).py"
]

for current_index in range(len(run_file_paths)):
    os.system(f"python -m {run_file_paths[current_index]}")

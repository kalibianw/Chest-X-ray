import os


file_path = "splited_Pneumonia_all_true_false_split_1_(300,300).py"
print(os.path.splitext(file_path)[0])
print(type(os.path.splitext(file_path)[0]))

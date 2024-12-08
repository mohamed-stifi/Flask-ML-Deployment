import os

def print_tree(directory, prefix=""):
    print(f"{prefix}{os.path.basename(directory)}/")
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            print_tree(item_path, prefix + "  ")
        else:
            print(f"{prefix}  {item}")

project_directory = "../app/"
print_tree(project_directory)

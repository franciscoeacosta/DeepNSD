import os, sys, re, stat, random
from copy import deepcopy
from pprint import pprint
from itertools import chain

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def flatten_list(lst):
    return [item for sublist in lst for item in sublist]
        
def split_list(lst, n):
    chunk_size = int(np.ceil(len(lst) / n))
    return list(chunk_list(lst, chunk_size))

def subset_complete_options(model_options, results_dir):
    results_files = [file.split('.')[0] for file in os.listdir(results_dir)]
    return [model_uid for model_uid in results_files
            if len(model_uid) > 0 and model_uid in model_options]

def subset_incomplete_options(model_options, results_dir):
    complete_models = subset_complete_options(model_options, results_dir)
    return [model_uid for model_uid in model_options
            if model_uid not in complete_models]
    
def write_commands_to_script(commands, output_shfile_name, output_shfile_dir = 'scriptoids', 
                             add_shuffle = False, bash_batch = None):
    if add_shuffle:
        random.shuffle(commands)

    if not os.path.exists(output_shfile_dir):
        os.makedirs(output_shfile_dir)

    if not bash_batch:
        output_shfile = '{}/{}.sh'.format(output_shfile_dir, output_shfile_name)
        with open(output_shfile, 'w') as file:
            file.write("\n".join(commands))

        os.chmod(output_shfile, os.stat(output_shfile).st_mode | stat.S_IEXEC)

    if bash_batch:
        chunk_size = int(np.ceil(len(commands) / bash_batch))

        for subset_i, command_subset in enumerate(list(chunk_list(commands, chunk_size))):
            output_shfile = '{}/{}{}.sh'.format(output_shfile_dir, output_shfile_name, subset_i)
            with open(output_shfile, 'w') as file:
                file.write("\n".join(command_subset))

            os.chmod(output_shfile, os.stat(output_shfile).st_mode | stat.S_IEXEC)
            
def find_most_nested_folders(paths):
    most_nested_folders = set()
    for path in paths:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]  # Ignore hidden folders
            if not dirs:  # Only add folder if it has no subdirectories
                most_nested_folders.add(root)
    return most_nested_folders

# written by ChatGPT4 
def find_missing_files(paths, remove_ext = True, filenames_only = True):
    all_files = {}
    missing_files = {}

    def traverse(folder_path):
        nonlocal all_files

        folder_files = set(os.listdir(folder_path))
        for entry in folder_files:
            entry_path = os.path.join(folder_path, entry)
            if os.path.isdir(entry_path) and not entry.startswith('.'):  # Ignore hidden folders
                traverse(entry_path)
            elif not entry.startswith('.'):  # Ignore hidden files
                all_files.setdefault(entry, []).append(entry_path)

    for path in paths:
        traverse(path)

    most_nested_folders = find_most_nested_folders(paths)

    for file, file_paths in all_files.items():
        file_folder_paths = set(os.path.dirname(file_path) for file_path in file_paths)
        missing_folders = most_nested_folders - file_folder_paths
        if missing_folders:
            for missing_folder in missing_folders:
                missing_path = os.path.join(missing_folder, file)
                missing_files.setdefault(file, []).append(missing_path)

    if remove_ext:
        missing_files = {file.split('.')[0]: missing_paths 
                         for file, missing_paths in missing_files.items()}
    
    if filenames_only:
        return list(missing_files.keys())
    
    return missing_files
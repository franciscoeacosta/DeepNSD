import pandas as pd
import numpy as np
import torch
import os, sys
import psutil
import subprocess
from xml import etree

from .memory import MemoryStats, convert_memory_unit

# Get CPU + GPU Info ------------------------------------------------------------------------

def get_cpu_info(pandas=True, accessible_keys=False, unit='GB', rowwise=True):
    total_memory = psutil.virtual_memory().total
    available_memory = psutil.virtual_memory().available
    total_cores = psutil.cpu_count(logical=True)
    available_cores = len(psutil.Process().cpu_affinity())
    
    ram_info = {
        'Total Memory': convert_memory_unit(total_memory, output_units=unit),
        'Free Memory': convert_memory_unit(available_memory, output_units=unit),
        'Free Memory (%)': round((available_memory / total_memory) * 100, 2),
        'Total Cores': total_cores,
        'Available Cores': available_cores,
    }
    
    cpu_utilization = psutil.cpu_percent(interval=1)
    ram_info['CPU Utilization'] = f"{cpu_utilization}% Utilized"

    if accessible_keys:
        accessible_names = {
            'Total Memory': 'total_memory',
            'Free Memory': 'free_memory',
            'Free Memory (%)': 'free_memory_percent',
            'Total Cores': 'total_cores',
            'Available Cores': 'available_cores',
            'CPU Utilization': 'cpu_utilization'
        }
        
        ram_info = {accessible_names[k]: v for k, v in ram_info.items()}
        
    pandas_output = pd.DataFrame([ram_info])
    
    if rowwise: # transpose the single row dataframe
        pandas_output = pandas_output.transpose()
        pandas_output.columns = [''] # set to empty

    return pandas_output if pandas else ram_info

def get_gpu_info(device_ids=None, pandas=True, simplify=True, 
                 accessible_keys=False, visible_devices=True,
                 utilization_data=False, average_utilization_over=3):
    
    accessible_names = {'GPU ID': 'gpu_id', 'Torch Device': 'device_name', 
                        'Utilization (%)': 'utilization', '# Processes': 'n_processes', 
                        'Free Memory (GB)': 'free_memory', 
                        'Used Memory (GB)': 'used_memory',
                        'Total Memory (GB)': 'total_memory',
                        'Free Memory (%)': 'free_memory_percent'}
                        
    key_outputs = ['GPU ID', 'Torch Device', 'Used Memory (MiB)', 'Total Memory (MiB)',
                   'Free Memory (%)', 'Utilization (%)', '# Processes']

    visible_devices = list(range(torch.cuda.device_count()))
    visible_devices_environ = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible_devices_environ is not None:
        visible_devices = list(map(int, visible_devices_environ.split(',')))
    
    def get_average_gpu_utilization():
        gpus = gpu_info_tree.findall("gpu")
        utilization_data = {i: [] for i in range(len(gpus))}
        
        start_time = time.time()
        while time.time() - start_time < average_utilization_over:
            gpu_infos = get_gpu_info()

            for i, gpu in enumerate(gpus):
                utilization = (gpu.find("utilization")
                               .find("gpu_util").text.split()[0])
                
                utilization_data[i].append(int(utilization))

            time.sleep(0.1)
        
        return {k: np.mean(v) for k,v in utilization_data.items()}
    
    command = "nvidia-smi -q -x"
    gpu_info_xml = subprocess.check_output(command, shell=True)
    gpu_info_tree = etree.ElementTree.fromstring(gpu_info_xml)

    gpu_infos, device_index = [], -1
    for gpu_i, gpu in enumerate(gpu_info_tree.findall("gpu")):
        if gpu_i in visible_devices:
            device_index += 1
            
            gpu_id, product_name = gpu.attrib['id'], gpu.find("product_name").text

            device_name, memory_info = f"cuda:{device_index}",  gpu.find("fb_memory_usage")
            used_memory = memory_info.find("used").text.split()[0]
            free_memory = memory_info.find("free").text.split()[0]
            total_memory = memory_info.find("total").text.split()[0]
            used_memory_percent = round(int(used_memory) / int(total_memory) * 100, 2)
            free_memory_percent = round(int(free_memory) / int(total_memory) * 100, 2)

            if device_ids is not None and gpu_id not in device_ids:
                continue 

            memory_free = gpu.find("fb_memory_usage").find("free").text.split()[0]

            processes = gpu.find("processes")
            num_processes = 0
            if processes is not None:
                num_processes = len(processes.findall("process_info"))

            gpu_infos.append({
                "GPU ID": gpu_id, 
                "GPU Name": product_name,
                "Torch Device": device_name,
                "Utilization (%)": '?',
                "Free Memory (%)": free_memory_percent,
                "Used Memory (%)": used_memory_percent,
                "Free Memory (GB)": free_memory,
                "Used Memory (GB)": used_memory,
                "Total Memory (GB)": total_memory,
                "# Processes": num_processes,
            })

        if utilization_data:
            utilization_data = get_average_gpu_utilization()
            for i in utilization_data:
                gpu_infos[i]['Utilization (%)'] = utilization_data[i]
            
    if simplify:
        gpu_infos = [{k:v for k,v in gpu_infos[i].items() 
                      if k in key_outputs} for i in range(len(gpu_infos))]
            
    if accessible_keys:
        gpu_infos = [{accessible_names[k]: np.nan if v=='?' else v 
                         for k,v in gpu_infos[i].items()
                         if k in accessible_names} 
                     for i in range(len(gpu_infos))]
            
    return pd.DataFrame(gpu_infos) if pandas else {i: gpu_infos[i] for i in range(len(gpu_infos))}

def get_gpu_info(device_ids=None, pandas=True, simplify=True, 
                 accessible_keys=False, visible_devices=True,
                 utilization_data=False, device_prefix='cuda', **kwargs):
    
    accessible_names = {'GPU ID': 'gpu_id', 'Device Name': 'device_name', 
                        'Utilization (%)': 'utilization', '# Processes': 'n_processes', 
                        'Free Memory': 'free_memory', 
                        'Used Memory': 'used_memory',
                        'Total Memory': 'total_memory',
                        'Free Memory (%)': 'free_memory_percent'}
                        
    key_outputs = ['Device Name', 'Used Memory', 'Total Memory', 'Free Memory', 
                   'Free Memory (%)', 'Utilization (%)', '# Processes']
    
    def get_average_gpu_utilization(**kwargs):
        average_utilization_over = kwargs.pop('average_utilization_over', 3)
        gpus = gpu_info_tree.findall("gpu")
        utilization_data = {i: [] for i in range(len(gpus))}
        
        start_time = time.time()
        while time.time() - start_time < average_utilization_over:
            gpu_infos = get_gpu_info()

            for i, gpu in enumerate(gpus):
                utilization = (gpu.find("utilization")
                               .find("gpu_util").text.split()[0])
                
                utilization_data[i].append(int(utilization))

            time.sleep(0.1)
        
        return {k: np.mean(v) for k,v in utilization_data.items()}
    
    command = "nvidia-smi -q -x"
    gpu_info_xml = subprocess.check_output(command, shell=True)
    gpu_info_tree = etree.ElementTree.fromstring(gpu_info_xml)
    
    check_visible_devices = True if visible_devices else False
    visible_devices = range(len(gpu_info_tree.findall("gpu")))
    visible_devices_setting = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible_devices_setting is not None and check_visible_devices:
        visible_devices = list(map(int, visible_devices_setting.split(',')))

    gpu_infos, device_id = [], -1
    for gpu_i, gpu in enumerate(gpu_info_tree.findall("gpu")):
        if gpu_i in visible_devices:
            device_id += 1
            
            gpu_id, product_name = gpu.attrib['id'], gpu.find("product_name").text

            device_name, memory_info = f"{device_prefix}:{device_id}", gpu.find("fb_memory_usage")
            used_memory = round(int(memory_info.find("used").text.split()[0]) / 1024, 2)
            free_memory = round(int(memory_info.find("free").text.split()[0]) / 1024, 2)
            total_memory = round(int(memory_info.find("total").text.split()[0]) / 1024, 2)
            used_memory_percent = round(int(used_memory) / int(total_memory) * 100, 2)
            free_memory_percent = round(int(free_memory) / int(total_memory) * 100, 2)

            if device_ids is not None and gpu_id not in device_ids:
                continue 

            memory_free = gpu.find("fb_memory_usage").find("free").text.split()[0]

            processes = gpu.find("processes")
            num_processes = 0
            if processes is not None:
                num_processes = len(processes.findall("process_info"))

            gpu_infos.append({
                "GPU ID": gpu_id, 
                "GPU Name": product_name,
                "Device Name": device_name,
                "Utilization (%)": '?',
                "Free Memory": str(free_memory)+' GB',
                "Used Memory": str(used_memory)+' GB',
                "Total Memory": str(total_memory)+' GB',
                "Free Memory (%)": free_memory_percent,
                "Used Memory (%)": used_memory_percent,
                "# Processes": num_processes,
            })

        if utilization_data:
            utilization_data = get_average_gpu_utilization()
            for i in utilization_data:
                gpu_infos[i]['Utilization (%)'] = utilization_data[i]
            
    if simplify:
        gpu_infos = [{k:v for k,v in gpu_infos[i].items() 
                      if k in key_outputs} for i in range(len(gpu_infos))]
            
    if accessible_keys:
        gpu_infos = [{accessible_names[k]: np.nan if v=='?' else v 
                         for k,v in gpu_infos[i].items()
                         if k in accessible_names} 
                     for i in range(len(gpu_infos))]
            
    return pd.DataFrame(gpu_infos) if pandas else {i: gpu_infos[i] for i in range(len(gpu_infos))}

def get_device_memory(pandas=True, accessible_keys=False, unit='GB', visible_devices=True):
    accessible_names = {'Device Name': 'device_name',
                        'Free Memory (%)': 'free_memory_percent',
                        'Free Memory': 'free_memory', 
                        'Total Memory': 'total_memory'}

    
    # Get CPU information
    cpu_info = get_cpu_info(pandas=False, accessible_keys=False)
    cpu_info['Device Name'] = 'cpu'
    cpu_info['Utilization (%)'] = cpu_info.pop('CPU Utilization', '?')
    cpu_info = {key: value for key, value in cpu_info.items() 
                    if key in accessible_names}
    
    info_list = [cpu_info]
    
    # Get GPU information
    gpu_infos = get_gpu_info(pandas=False, accessible_keys=False, 
                             visible_devices=visible_devices)
    
    for gpu_index, gpu_info in gpu_infos.items():
        info_list.append({key: value for key, value in gpu_info.items() 
                              if key in accessible_names})
    
    device_info_list = []
    for info in info_list:
        parsed_info = {}
        for key, value in info.items():
            output_key = accessible_names[key] if accessible_keys else key
            parsed_info[output_key] = value
            if 'Memory' in key and '%' not in key:
                parsed_info[output_key] = convert_memory_unit(value, output_units=unit)
                
        device_info_list.append(parsed_info)
                
    if pandas:
        return pd.DataFrame(device_info_list)[list(accessible_names.keys())]

    return {info.pop('device_name'): info for info in device_info_list}

class SystemStats:
    def __init__(self):
        self.cpu_info = get_cpu_info(pandas=True, accessible_keys=True)
        self.gpu_info = get_gpu_info(pandas=True, accessible_keys=True)
        
        self.available_cores = self.cpu_info.loc['total_cores'][0]
        self.available_ram = self.cpu_info.loc['free_memory'][0]
        self.available_gpu = {'n_gpus': len(self.gpu_info),
                              'memory': MemoryStats().sum(self.gpu_info['free_memory'].tolist())}
        
    def __repr__(self):
        return self.get_report()
        
    def get_report(self):
        return (f"Available RAM: {self.available_ram} \nAvailable Cores: {self.available_cores} \n" +
                f"Available GPU(s): {self.available_gpu['n_gpus']}: {self.available_gpu['memory']}")


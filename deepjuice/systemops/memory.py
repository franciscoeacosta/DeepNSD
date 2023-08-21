import pandas as pd
import numpy as np
import torch
import os, sys
import psutil
import subprocess
from xml import etree

# Setup: Units + Conversion ------------------------------------------------------

def split_unit_from_numeric(value, assume_int=False):
    if not isinstance(value, str):
        raise ValueError('str value expected')
    for i in range(len(value)):
        if value[i].isalpha():
            break
    number, units = value[:i], value[i:]

    if '.' in number or not assume_int:
        return float(number), units
    
    return int(number), units

def reformat_bytes_unit(bytes_unit, unit_format='acronym_lower'):
    base_lookup = {
        'b': '',
        'k': 'kilo',
        'm': 'mega',
        'g': 'giga',
        't': 'tera',
        'p': 'peta',
        'e': 'eta',
        'z': 'zetta',
        'y': 'yotta',
    }
    
    # Ensure bytes_unit is lowercase and has at least 2 characters
    bytes_unit = bytes_unit.lower()
    
    if len(bytes_unit) == 1:
        if bytes_unit.upper() != 'B':
            bytes_unit = bytes_unit + 'B'
    
    # Determine the base of the unit (e.g., 'kilo' for 'kb')
    base = base_lookup.get(bytes_unit[0], '')
    
    # Map each unit_format to the appropriate representation
    formats = {'lower': base + 'bytes', 'title': base.title() + 'Bytes',
               'acronym_lower': bytes_unit, 'acronym_upper': bytes_unit.upper()}
    
    if unit_format not in formats:
        raise ValueError(f"Invalid unit_format. Must be one of {list(formats.keys())}")
    
    return formats[unit_format]

def reformat_memory_unit(memory, unit='auto', digits='auto:3', tight=False):
    rounding = {'B': 0, 'KB': 1, 'MB': 2, 'GB': 3, 'TB': 4, 'PB': 5, 'EB': 6, 'ZB': 7, 'YB': 8}
    
    if unit is None or unit == 'auto':
        memory, unit = split_unit_from_numeric(bytes_output)
        
    if digits is None:
        return memory
    
    parsed_unit = reformat_bytes_unit(unit, unit_format='acronym_upper')
    
    if parsed_unit not in rounding:
        raise ValueError(f"Invalid unit: {unit}")
    
    if not isinstance(digits, int) and 'auto' in digits:
        auto_digits = rounding[parsed_unit]
        max_digits = auto_digits
        if ':' in digits:
            max_digits = int(digits.split(':')[-1])
        
        digits = min(auto_digits, max_digits)
        
    memory_in_units = f"{memory:.{digits}f} {unit}"
    if tight: memory_in_units = memory_in_units.replace(' ', '')
    
    return memory_in_units
    
def convert_memory_unit(memory, input_units='B', output_units='auto', readable=True, 
                        format_kwargs={'digits': 'auto:3', 'tight': False}):
                          
    exponents = {'B': 0, 'KB': 1, 'MB': 2, 'GB': 3, 'TB': 4, 'PB': 5, 'EB': 6, 'ZB': 7, 'YB': 8}
    unit_dict = {v: k for k,v in exponents.items()}
    
    def convert_to_bytes(value, unit):
        unit = reformat_bytes_unit(unit, unit_format='acronym_upper')
        converted_value = value * (1024 ** exponents[unit])
        return int(converted_value) if unit == 'B' else converted_value

    # Function to convert bytes to the desired memory units
    def convert_from_bytes(value, unit):
        unit = reformat_bytes_unit(unit, unit_format='acronym_upper')
        converted_value = value / (1024 ** exponents[unit])
        return round(converted_value) if unit == 'B' else converted_value
    
    def determine_memory_unit(x):
        if x == 0: return 'B'
        magnitude = int(np.log2(x) / 10)
        if magnitude > 8: 
            magnitude = 8
        return unit_dict[magnitude]

    # If the memory is given as a string, split it into the number and units
    if isinstance(memory, str):
        memory, input_units = split_unit_from_numeric(memory)

    # Convert the memory to bytes
    memory_in_bytes = convert_to_bytes(memory, input_units)
    
    if output_units == 'auto':
        output_units = determine_memory_unit(memory_in_bytes)

    # Convert the memory from bytes to the desired output units
    memory_in_output_units = convert_from_bytes(memory_in_bytes, output_units)
    
    if not readable:
        return float(memory_in_output_units)
    
    return reformat_memory_unit(memory_in_output_units, output_units, **format_kwargs)

# Memory Usage Functions ------------------------------------------------------

def compute_size_in_memory(data, units='B', readable=False):
    def get_memory_use(x):
        if isinstance(x, torch.Tensor):
            return x.element_size() * x.nelement()
        elif isinstance(x, np.ndarray):
            return x.nbytes
        return sys.getsizeof(x)

    valid_dtypes = [int, float, str, dict, list, np.array, torch.Tensor]

    if isinstance(data, (np.ndarray, torch.Tensor, int, float, str)):
        memory_usage = get_memory_use(data)
        
    elif isinstance(data, (list, dict)):
        memory_usage = 0
        if isinstance(data, list):
            for value in data:
                memory_usage += get_memory_use(value)
        
        if isinstance(data, dict):
            for key, value in data.items():
                memory_usage += get_memory_use(key) + get_memory_use(value)
            
    else:
        raise ValueError(f'Unknown data type, must be one of {valid_dtypes}')
    
    return convert_memory_unit(memory_usage, output_units=units, readable=readable)

def get_available_device_memory(device='cpu'):
    if device == 'cpu' or device is None:
        return psutil.virtual_memory().available

    if 'cuda' in device or 'gpu' in device:
        device_index = (0 if ':' not in device else 
                        int(device.split(':')[-1]))
        
        command = "nvidia-smi -q -x"
        gpu_info_xml = subprocess.check_output(command, shell=True)
        gpu_info_tree = etree.ElementTree.fromstring(gpu_info_xml)

        gpu = gpu_info_tree.findall("gpu")[device_index]
        free_memory_mbytes = (int(gpu.find("fb_memory_usage")
                               .find("free").text.split()[0]))
        
        return free_memory_mbytes * 1024 * 1024
    raise ValueError("Invalid device. Must be 'cpu' or 'cuda/gpu:{device_index}'")

# MemoryStats Helper Class ----------------------------------------------------------

class MemoryStats:
    def __init__(self, data=None, index=None, output_units='auto', readable=True):
        self.output_units = output_units
        self.readable = readable
        
        self.exponents = {'B': 0, 'KB': 1, 'MB': 2, 'GB': 3, 
                          'TB': 4, 'PB': 5, 'EB': 6, 'ZB': 7, 'YB': 8}
        
        self.unit_dict = {value: key for key, value in self.exponents.items()}
        
        if data is not None:
            self.data = self.data_to_series(data, index)
            
        for alt_names in ['get_memory', 'compute_memory', 'get_size', 'get_size_in_memory']:
            setattr(self, alt_names, self.compute_memory_usage)
            
    @staticmethod
    def data_to_series(data, index=None):
        if isinstance(data, (pd.Series, np.number, np.ndarray, list)):
            return pd.Series(data, index)
        if isinstance(data, dict):
            return pd.Series(list(data.values()), index)
        if isinstance(data, torch.Tensor):
            return pd.Series(data.numpy(), index)
        if isinstance(data, DataLoader):
            return pd.Series(np.array([batch.numpy() for batch in data]), index)
        raise TypeError(f"Invalid input type: {type(data)}")
        
    def convert_from_bytes(self, x, input_units='B', **kwargs):
        convert_args = [input_units, self.output_units, self.readable]
        if isinstance(x, (int, float, np.number)):
            return convert_memory_unit(x, *convert_args, **kwargs)
        if isinstance(x, pd.Series):
            return x.apply(convert_memory_unit, args = tuple(convert_args), **kwargs)
        if isinstance(x, list):
            return [convert_memory_unit(v, *convert_args, **kwargs) for v in x]
        if isinstance(x, dict):
            return {k: convert_memory_unit(v, *convert_args, **kwargs) for k,v in x.items()}
        raise TypeError(f"Invalid type for conversion: {type(x)}")
        
    def determine_unit(self, x):
        if x == 0: return 'B'
        magnitude = int(np.log2(x) / 10)
        if magnitude > 8: 
            magnitude = 8
        return self.unit_dict[magnitude]
    
    def compute_memory_usage(self, x=None, units=None, readable=None):
        if x is None:
            x = getattr(self, 'data', None)
        if x is None:
            raise ValueError('please provide argument: {x} for memory estimation.')
            
        if readable is None:
            readable = self.readable
            
        if units is None:
            units = self.output_units
            
        if isinstance(x, (torch.Tensor, np.ndarray)):
            return compute_size_in_memory(x, units, readable)
        
        return convert_memory_unit(self.sum(x), output_units=units, readable=readable) 
            
    def _convert_for_stats(self, x):
        return convert_memory_unit(x, output_units='B', readable=False)

    def _convert_from_stat(self, x):
        return convert_memory_unit(x, output_units=self.output_units, readable=self.readable)

    def _stat_method(self, data, method):
        data_in_bytes = self.data_to_series(data).apply(self._convert_for_stats)
        result = getattr(data_in_bytes, method)()
        if isinstance(result, pd.Series):
            return result.apply(self._convert_from_stat)
        else:
            return self._convert_from_stat(result)

    def __getattr__(self, method):
        if method in pd.Series.__dict__:
            return lambda x: self._stat_method(x, method)
        raise AttributeError(f"'MemoryOpts' object has no attribute '{method}'")

# MemorySeries (pd.Series Mod) ------------------------------------------------------

class MemorySeries(pd.Series):
    def __init__(self, data=None, index=None, dtype=None, 
                 name=None, copy=False, fastpath=False, output_units='auto'):
        
        super().__init__(data, index, dtype, name, copy, fastpath)
        
        self.output_units = output_units
        
        self.exponents = {'B': 0, 'KB': 1, 'MB': 2, 'GB': 3, 
                          'TB': 4, 'PB': 5, 'EB': 6, 'ZB': 7, 'YB': 8}
        
        self.unit_dict = {value: key for key, value in self.exponents.items()}
        
        methods_to_override = ['sum', 'mean', 'median', 'min', 'max', 'mode',
                               'var', 'std', 'skew', 'kurt', 'quantile', 
                               'cumsum', 'cummin', 'cummax', 'cumprod',
                               'describe','count', 'nunique','value_counts']
        
        for method in methods_to_override:
            setattr(MemorySeries, method, lambda self, method=method: self._stat_method(method))
            
        self.stats_methods = methods_to_override + ['cummean']
        
    @staticmethod
    def _convert_for_stat(x):
        return convert_memory_unit(x, output_units='B', readable=False)
    
    def determine_unit(self, x):
        if x == 0: return 'B'
        magnitude = int(np.log2(x) / 10)
        if magnitude > 8: 
            magnitude = 8
        return self.unit_dict[magnitude]

    def _convert_from_stat(self, x):
        if self.output_units == 'auto':
            unit = self._determine_unit(x)
        else:
            unit = self.output_units
        return convert_memory_unit(x, output_units=unit, readable=True)

    # A generalized statistical method
    def _stat_method(self, method):
        x_bytes = super().apply(lambda x: self._convert_for_stat(x))
        result = getattr(x_bytes, method)()
        if isinstance(result, pd.Series):
            return result.apply(self._convert_from_stat)
        else:
            return self._convert_from_stat(result)
        
    def cummean(self):
        cumsum_in_bytes = super().apply(lambda x: self._convert_for_stat(x)).cumsum()
        count = pd.Series(range(1, len(self) + 1), index=self.index)
        return (cumsum_in_bytes / count).apply(self._convert_from_stat)
    
def convert_to_memory_series(series):
    return MemorySeries(series.apply(convert_memory_unit).tolist(), index = series.index)

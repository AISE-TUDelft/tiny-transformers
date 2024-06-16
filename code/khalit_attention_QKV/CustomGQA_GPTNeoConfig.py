import torch
from torch import nn
from transformers import GPTNeoForCausalLM
from transformers import GPTNeoConfig
import json
from typing import Any, Dict

class CustomGPTNeoConfig(GPTNeoConfig):
    model_type = "custom_gqa_gpt_neo"
    def __init__(self, num_kv_heads=None, kqv_size=None, **kwargs):
        if num_kv_heads is None:
            print(kwargs)
            self.num_kv_heads = kwargs['num_heads']
        else:
            self.num_kv_heads = num_kv_heads
        if kqv_size is None:
            self.kqv_size = kwargs['hidden_size']
        else:
            self.kqv_size = kqv_size
        super().__init__(**kwargs)

    def to_dict(self):
        # Get the dictionary representation of the base class
        config_dict = super().to_dict()
        
        # Add custom attributes
        config_dict['num_kv_heads'] = self.num_kv_heads
        config_dict['kqv_size'] = self.kqv_size
        
        return config_dict
    
    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
    
    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = GPTNeoConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (
                key not in default_config_dict
                or value != default_config_dict[key]
            ):
                serializable_config_dict[key] = value

        return serializable_config_dict
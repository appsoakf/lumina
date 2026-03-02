import json
import os


def load_module_config(module_file: str) -> dict:
    """加载与调用模块同目录的 config.json。"""
    config_path = os.path.join(os.path.dirname(module_file), 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

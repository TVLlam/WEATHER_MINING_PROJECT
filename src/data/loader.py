"""
Utility module for loading project configuration.
Reads parameters from configs/params.yaml.
"""

import os
import yaml


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from a YAML file.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the YAML config file. If None, defaults to
        'configs/params.yaml' relative to the project root.
    
    Returns
    -------
    dict
        Dictionary containing all configuration parameters.
    """
    if config_path is None:
        # Tìm project root (nơi chứa thư mục configs/)
        project_root = _find_project_root()
        config_path = os.path.join(project_root, "configs", "params.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def _find_project_root() -> str:
    """
    Tìm thư mục gốc của project bằng cách tìm ngược từ vị trí hiện tại
    lên thư mục cha cho đến khi tìm thấy thư mục 'configs/'.
    """
    current = os.path.abspath(os.path.dirname(__file__))
    
    # Duyệt ngược lên tối đa 5 cấp
    for _ in range(5):
        if os.path.isdir(os.path.join(current, "configs")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    
    # Fallback: dùng current working directory
    return os.getcwd()


def resolve_path(relative_path: str, config: dict = None) -> str:
    """
    Chuyển đổi đường dẫn tương đối thành đường dẫn tuyệt đối
    dựa trên project root.
    
    Parameters
    ----------
    relative_path : str
        Đường dẫn tương đối từ project root.
    config : dict, optional
        Config dict (không dùng, để mở rộng sau).
    
    Returns
    -------
    str
        Đường dẫn tuyệt đối.
    """
    project_root = _find_project_root()
    return os.path.join(project_root, relative_path)
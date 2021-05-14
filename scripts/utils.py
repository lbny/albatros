"""
Author: Lucas Bony

Utils functions
"""

def format_filepath(filepath: str, run_name: str) -> str:
    """Appends run_name to filepath"""
    if run_name == '':
        return filepath
    else:
        return '_'.join([run_name, filepath])
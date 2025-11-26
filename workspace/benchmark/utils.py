import os
from tdc.utils import retrieve_dataset_names

def get_dataset_group(dataset_name):
    """
    Determine if the dataset belongs to ADME or Tox.
    """
    dataset_name_lower = dataset_name.lower()
    
    try:
        tox_datasets = retrieve_dataset_names('Tox')
        # Check case-insensitive
        if dataset_name_lower in [d.lower() for d in tox_datasets]:
            return 'Tox'
    except:
        pass
        
    try:
        adme_datasets = retrieve_dataset_names('ADME')
        if dataset_name_lower in [d.lower() for d in adme_datasets]:
            return 'ADME'
    except:
        pass
    
    return None

def get_dataset_path(base_dir, dataset_name):
    """
    Get the path to the dataset directory.
    """
    # Try direct path
    direct_path = os.path.join(base_dir, dataset_name)
    if os.path.exists(direct_path):
        return direct_path

    # Try subdirectories
    group = get_dataset_group(dataset_name)
    if group:
        return os.path.join(base_dir, group, dataset_name)
    
    # Try searching in known groups if get_dataset_group failed (e.g. for custom names)
    for g in ['Tox', 'ADME']:
        p = os.path.join(base_dir, g, dataset_name)
        if os.path.exists(p):
            return p
            
    # Fallback
    return os.path.join(base_dir, dataset_name)

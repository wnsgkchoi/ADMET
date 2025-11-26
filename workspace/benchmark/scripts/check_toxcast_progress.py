
import os
import sys
from tdc.utils import retrieve_label_name_list

def check_progress():
    name = 'toxcast'
    try:
        labels = retrieve_label_name_list(name)
    except Exception as e:
        print(f"Error retrieving labels: {e}")
        return

    processed_dir = '/home/flash/workspace/ADMET/workspace/benchmark/data/Tox'
    if not os.path.exists(processed_dir):
        print("Processed directory not found.")
        return

    processed_folders = os.listdir(processed_dir)
    processed_labels = []
    
    # Extract label names from folder names
    # Folder format: toxcast_{label}
    prefix = "toxcast_"
    for folder in processed_folders:
        if folder.startswith(prefix):
            label = folder[len(prefix):]
            processed_labels.append(label)
    
    processed_set = set(processed_labels)
    all_labels_set = set(labels)
    
    missing = all_labels_set - processed_set
    
    print(f"Total labels in toxcast: {len(labels)}")
    print(f"Processed labels: {len(processed_set)}")
    print(f"Missing labels: {len(missing)}")
    
    if missing:
        print("First 10 missing labels:")
        for l in list(missing)[:10]:
            print(l)

if __name__ == "__main__":
    check_progress()

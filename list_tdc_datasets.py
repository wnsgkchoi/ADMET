from tdc.utils import retrieve_dataset_names

# TDC categories
groups = ['ADME', 'Tox'] 

for group in groups:
    print(f"Group: {group}")
    try:
        names = retrieve_dataset_names(group)
        for name in names:
            print(f"  - {name}")
    except Exception as e:
        print(f"  Error retrieving {group}: {e}")

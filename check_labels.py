from tdc.utils import retrieve_label_name_list

datasets = ['tox21', 'toxcast', 'ames']
for d in datasets:
    print(f"Dataset: {d}")
    try:
        labels = retrieve_label_name_list(d)
        print(f"  Labels: {labels}")
    except Exception as e:
        print(f"  Error: {e}")

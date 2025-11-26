
# ===================================================================
# 1. ADMET 카테고리별 데이터셋 정의
# ===================================================================

# Absorption (8개)
ABSORPTION_DATASETS = {
    'Caco2_Wang': {'type': 'regression', 'unit': 'log cm/s', 'description': 'Caco-2 permeability', 'loader': 'ADME'},
    'Lipophilicity_AstraZeneca': {'type': 'regression', 'unit': 'log D', 'description': 'Lipophilicity', 'loader': 'ADME'},
    'Solubility_AqSolDB': {'type': 'regression', 'unit': 'log mol/L', 'description': 'Aqueous solubility', 'loader': 'ADME'},
    'HydrationFreeEnergy_FreeSolv': {'type': 'regression', 'unit': 'kcal/mol', 'description': 'Hydration free energy', 'loader': 'ADME'},
    'HIA_Hou': {'type': 'classification', 'description': 'Human Intestinal Absorption', 'loader': 'ADME'},
    'Pgp_Broccatelli': {'type': 'classification', 'description': 'P-glycoprotein inhibition', 'loader': 'ADME'},
    'Bioavailability_Ma': {'type': 'classification', 'description': 'Oral bioavailability', 'loader': 'ADME'},
    'PAMPA_NCATS': {'type': 'classification', 'description': 'PAMPA permeability', 'loader': 'ADME'}
}

# Distribution (3개)
DISTRIBUTION_DATASETS = {
    'BBB_Martins': {'type': 'classification', 'description': 'Blood-Brain Barrier penetration', 'loader': 'ADME'},
    'PPBR_AZ': {'type': 'regression', 'unit': '%', 'description': 'Plasma Protein Binding Rate', 'loader': 'ADME'},
    'VDss_Lombardo': {'type': 'regression', 'unit': 'L/kg', 'description': 'Volume of Distribution', 'loader': 'ADME'}
}

# Metabolism (8개)
METABOLISM_DATASETS = {
    'CYP2C19_Veith': {'type': 'classification', 'description': 'CYP2C19 inhibition', 'loader': 'ADME'},
    'CYP2D6_Veith': {'type': 'classification', 'description': 'CYP2D6 inhibition', 'loader': 'ADME'},
    'CYP3A4_Veith': {'type': 'classification', 'description': 'CYP3A4 inhibition', 'loader': 'ADME'},
    'CYP1A2_Veith': {'type': 'classification', 'description': 'CYP1A2 inhibition', 'loader': 'ADME'},
    'CYP2C9_Veith': {'type': 'classification', 'description': 'CYP2C9 inhibition', 'loader': 'ADME'},
    'CYP2C9_Substrate_CarbonMangels': {'type': 'classification', 'description': 'CYP2C9 substrate', 'loader': 'ADME'},
    'CYP2D6_Substrate_CarbonMangels': {'type': 'classification', 'description': 'CYP2D6 substrate', 'loader': 'ADME'},
    'CYP3A4_Substrate_CarbonMangels': {'type': 'classification', 'description': 'CYP3A4 substrate', 'loader': 'ADME'}
}

# Excretion (3개)
EXCRETION_DATASETS = {
    'Half_Life_Obach': {'type': 'regression', 'unit': 'hour', 'description': 'Half life', 'loader': 'ADME'},
    'Clearance_Hepatocyte_AZ': {'type': 'regression', 'unit': 'mL/min/kg', 'description': 'Clearance (Hepatocyte)', 'loader': 'ADME'},
    'Clearance_Microsome_AZ': {'type': 'regression', 'unit': 'mL/min/kg', 'description': 'Clearance (Microsome)', 'loader': 'ADME'}
}

# Toxicity (11개: 8개 + hERG Central 3개)),  *주의* hERG Central, Tox21, ToxCast는 멀티 레이블
TOXICITY_DATASETS = {
    'LD50_Zhu': {'type': 'regression', 'unit': 'log(1/(mol/kg))', 'description': 'Acute toxicity LD50', 'loader': 'Tox'},
    'hERG': {'type': 'classification', 'description': 'hERG blockers', 'loader': 'Tox'},
    'hERG_Karim': {'type': 'classification', 'description': 'hERG blockers (Karim)', 'loader': 'Tox'},
    'AMES': {'type': 'classification', 'description': 'Ames mutagenicity', 'loader': 'Tox'},
    'DILI': {'type': 'classification', 'description': 'Drug-Induced Liver Injury', 'loader': 'Tox'},
    'Skin_Reaction': {'type': 'classification', 'description': 'Skin reaction', 'loader': 'Tox'},
    'Carcinogens_Lagunin': {'type': 'classification', 'description': 'Carcinogenicity', 'loader': 'Tox'},
    'ClinTox': {'type': 'classification', 'description': 'Clinical trial toxicity', 'loader': 'Tox'},

     # hERG Central
    'hERG_Central_1uM': {
        'type': 'regression',
        'unit': '% inhibition',
        'description': 'hERG inhibition at 1uM',
        'loader': 'Tox',
        'dataset_name': 'herg_central',
        'label_name': 'hERG_at_1uM'
    },
    'hERG_Central_10uM': {
        'type': 'regression',
        'unit': '% inhibition',
        'description': 'hERG inhibition at 10uM',
        'loader': 'Tox',
        'dataset_name': 'herg_central',
        'label_name': 'hERG_at_10uM'
    },
    'hERG_Central_inhib': {
        'type': 'classification',
        'description': 'hERG inhibition (binary)',
        'loader': 'Tox',
        'dataset_name': 'herg_central',
        'label_name': 'hERG_inhib'
    }
}

ALL_TARGET_DATASETS = {}
ALL_TARGET_DATASETS.update(ABSORPTION_DATASETS)
ALL_TARGET_DATASETS.update(DISTRIBUTION_DATASETS)
ALL_TARGET_DATASETS.update(METABOLISM_DATASETS)
ALL_TARGET_DATASETS.update(EXCRETION_DATASETS)
ALL_TARGET_DATASETS.update(TOXICITY_DATASETS)

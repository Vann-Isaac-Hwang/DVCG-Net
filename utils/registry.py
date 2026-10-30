# utils/registry.py
"""
全局数据集注册表
统一管理 CPSC / PTB-XL / Chapman / PTB 的路径和类别信息。
"""

import os

DATA_ROOT = os.environ.get(
    "DUAL_ECG_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
)

PTBXL_RAW_DIR = os.path.join(
    DATA_ROOT,
    "PTBXL_Data",
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
)

DATASET_REGISTRY = {
    "CPSC": {
        "num_classes": 9,
        "class_names": ["N", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"],
        "is_multilabel": True,
        "default_data_dir": os.path.join(DATA_ROOT, "CPSC_Data", "processed_sr100_freq10"),
        "raw_data_dir": os.path.join(DATA_ROOT, "CPSC_Data"),
        "csv_path": os.path.join(DATA_ROOT, "CPSC_Data", "REFERENCE.csv"),
    },

    "PTBXL": {
        "num_classes": 5,
        "class_names": ["NORM", "MI", "STTC", "CD", "HYP"],
        "is_multilabel": True,
        "default_data_dir": os.path.join(DATA_ROOT, "PTBXL_Data", "processed_sr100_freq10"),
        "raw_data_dir": PTBXL_RAW_DIR,
        "csv_path": os.path.join(PTBXL_RAW_DIR, "ptbxl_database.csv"),
        "scp_path": os.path.join(PTBXL_RAW_DIR, "scp_statements.csv"),
        "records100_dir": os.path.join(PTBXL_RAW_DIR, "records100"),
        "records500_dir": os.path.join(PTBXL_RAW_DIR, "records500"),
    },

    "CHAPMAN": {
        "num_classes": 4,
        "class_names": ["SB", "AFIB", "GSVT", "SR"],
        "is_multilabel": False,
        "default_data_dir": os.path.join(
            DATA_ROOT, "Chapman_Data", "processed_sr100_freq10"
        ),
        "raw_data_dir": os.path.join(DATA_ROOT, "Chapman_Data", "raw"),
        "metadata_path": os.environ.get(
            "CHAPMAN_FOLDS_PATH",
            os.path.join(DATA_ROOT, "Chapman_Data", "chapman_folds.csv"),
        ),
        "folds_path": os.environ.get(
            "CHAPMAN_FOLDS_PATH",
            os.path.join(DATA_ROOT, "Chapman_Data", "chapman_folds.csv"),
        ),
    },

    "PTB": {
        "num_classes": 2,
        "class_names": ["Healthy Control (HC)", "Myocardial Infarction (MI)"],
        "is_multilabel": False,
        "default_data_dir": os.path.join(DATA_ROOT, "PTB_Data", "processed"),
        "raw_data_dir": os.path.join(DATA_ROOT, "PTB_Data"),
    }
}


def get_dataset_info(dataset_name):
    dataset_name = dataset_name.upper()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"未知的数据集: {dataset_name}. 支持的选项: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]

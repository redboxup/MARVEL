import dataclasses
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class DatasetConf:
    name: str = dataclasses.field(init=False)
    labels: list[str] = dataclasses.field(init=False)
    labels_ood: list[str] = dataclasses.field(init=False)
    labels_fullform: Tuple[str, ...] = dataclasses.field(init=False)
    num_classes: int = dataclasses.field(init=False)
    head: int = dataclasses.field(init=False)
    mid: int = dataclasses.field(init=False)
    tail: int = dataclasses.field(init=False)
    nearood1: str = dataclasses.field(init=False)
    nearood2: str = dataclasses.field(init=False)
    nearood3: str = dataclasses.field(init=False)
    task: str = dataclasses.field(init=False)
    calibrate_using_head: bool = dataclasses.field(init=False)


@dataclass
class ISICDatasetConf(DatasetConf):
    name: str = "isic2019"
    train_csv_path: str = (
        "data/preprocessed/isic2019/"
        "ISIC_2019_Training_GroundTruth_split.csv"
    )
    val_csv_path: str = (
        "data/preprocessed/isic2019/ISIC_2019_Val_GroundTruth_split.csv"
    )
    test_csv_path: str = (
        "data/preprocessed/isic2019/ISIC_2019_Test_GroundTruth.csv"
    )
    train_image_path: str = (
        "data/preprocessed/isic2019/ISIC_2019_Training_Input/"
        "ISIC_2019_Training_Input"
    )
    val_image_path: str = (
        "data/preprocessed/isic2019/ISIC_2019_Training_Input/"
        "ISIC_2019_Training_Input"
    )
    test_image_path: str = (
        "data/preprocessed/isic2019/ISIC_2019_Test_Input/"
        "ISIC_2019_Test_Input"
    )
    num_classes: int = 6
    labels: list[str] = dataclasses.field(
        default_factory=lambda: [
            "MEL", "BCC", "BKL", "AK", "DF", "VASC",
        ]
    )
    labels_ood: list[str] = dataclasses.field(
        default_factory=lambda: [
            "NV", "MEL",
        ]
    )
    labels_fullform: Tuple[str, ...] = (
        "Melanocytic Nevus", "Melanoma", "Basal Cell Carcinoma",
        "Benign Keratosis", "Actinic Keratosis", "Squamous Cell Carcinoma",
        "Dermatofibroma", "Vascular Lesion",
    )
    head: int = 1
    mid: int = 3
    tail: int = 5

    nearood1: str = "dfu"
    nearood2: str = "padufes"
    nearood3: str = "isic2019_corr"
    task: str = "multiclass"
    calibrate_using_head: bool = True


@dataclass
class NCTCRCDatasetConf(DatasetConf):
    name: str = "nctcrc"
    num_classes: int = 6
    train_csv_path: str = "data/preprocessed/nctcrc/train_labels.csv"
    val_csv_path: str = "data/preprocessed/nctcrc/val_labels.csv"
    test_csv_path: str = "data/preprocessed/nctcrc/test_labels.csv"
    train_image_path: str = "data/preprocessed/nctcrc/"
    val_image_path: str = "data/preprocessed/nctcrc/"
    test_image_path: str = "data/preprocessed/nctcrc/"
    labels: list[str] = dataclasses.field(
        default_factory=lambda: [
            'TUM', 'MUS', 'LYM', 'STR', 'ADI', 'MUC',
        ]
    )
    labels_ood: list[str] = dataclasses.field(
        default_factory=lambda: [
            'BACK', 'DEB', 'NORM',
        ]
    )
    head: int = 1
    mid: int = 3
    tail: int = 5

    nearood1: str = "cytology"
    nearood2: str = "mihic"
    nearood3: str = "nctcrc_corr"
    task: str = "multiclass"
    calibrate_using_head: bool = False


@dataclass
class RFMiDDatasetConf(DatasetConf):
    name: str = "rfmid"
    num_classes: int = 16
    train_csv_path: str = "data/preprocessed/rfmid/RFMiD_Training_Labels.csv"
    val_csv_path: str = "data/preprocessed/rfmid/RFMiD_Validation_Labels.csv"
    test_csv_path: str = "data/preprocessed/rfmid/RFMiD_Testing_Labels.csv"
    train_image_path: str = (
        "data/preprocessed/rfmid/Training_Set/Training_Set/Training/"
    )
    val_image_path: str = (
        "data/preprocessed/rfmid/Evaluation_Set/Evaluation_Set/Validation/"
    )
    test_image_path: str = "data/preprocessed/rfmid/Test_Set/Test_Set/Test"
    labels: list[str] = dataclasses.field(
        default_factory=lambda: [
            'MH', 'DR', 'ODC', 'DN', 'ARMD', 'BRVO', 'ODE', 'MYA', 'RS',
            'CSR', 'CRVO', 'ODP', 'RT', 'EDN', 'MHL', 'MS',
        ]
    )
    labels_ood: list[str] = dataclasses.field(
        default_factory=lambda: [
            'TSLN', 'LS', 'ST', 'RP', 'CWS', 'CB', 'ODPM', 'PRH', 'MNF',
            'HR', 'CRAO', 'TD', 'CME', 'PTCR', 'CF', 'VH', 'MCA', 'VS',
            'BRAO', 'PLQ', 'HPED', 'CL', 'TV', 'PT', 'CRS', 'RPEC',
            'AION', 'ERM', 'AH',
        ]
    )
    head: int = 2
    mid: int = 8
    tail: int = 15

    nearood1: str = "deepdrid"
    nearood2: str = "oct"
    nearood3: str = "rfmid_corr"
    task: str = "multiclass"
    calibrate_using_head: bool = False


@dataclass
class OODDatasetConf(DatasetConf):
    aux: str = "imagenet100"
    farood: str = "mscoco5k"
    nearood1: str = ""
    nearood2: str = ""
    nearood3: str = ""

    imagenet100: str = "data/preprocessed/imagenet100"
    mscoco5k: str = "data/preprocessed/mscoco5k"
    dfu: str = "data/preprocessed/nearood/isic2019/DFU/DFU/"
    padufes: str = "data/preprocessed/nearood/isic2019/padufes/"
    deepdrid: str = (
        "data/preprocessed/nearood/rfmid/DeepDRiD/ultra-widefield_images"
    )
    oct: str = (
        "data/preprocessed/nearood/rfmid/oct/oct2017/test"
    )
    irma: str = (
        "data/preprocessed/nearood/nihcxr/irma/"
        "irma_proprocessed_without_ID"
    )
    mura: str = "data/preprocessed/nearood/nihcxr/mura/MURA-v1.1/train"
    cytology: str = "data/preprocessed/nearood/nctcrc/cytology"
    mihic: str = "data/preprocessed/nearood/nctcrc/mihic/val"
    isic2019_corr: str = "data/preprocessed/nearood/isic2019/corr"
    rfmid_corr: str = "data/preprocessed/nearood/rfmid/corr"
    nihcxr_corr: str = "data/preprocessed/nearood/nihcxr/corr"
    nctcrc_corr: str = "data/preprocessed/nearood/nctcrc/corr"

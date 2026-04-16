import cv2
import shutil
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Any
from pathlib import Path
from tqdm.auto import tqdm
from functools import partial
from multiprocessing import Pool
from torchvision.transforms import functional as F
from PIL import Image
from sklearn.model_selection import train_test_split


def crop_rfmid_img(img: Any):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # to pil
    if img.size == (4288, 2848):
        cropped_img = F.center_crop(img, [2848, 3800])
    elif img.size == (2048, 1536):
        cropped_img = F.center_crop(img, [1500, 1500])
    elif img.size == (2144, 1424):
        cropped_img = F.center_crop(img, [1400, 1400])
    else:
        raise ValueError(f"Unexpected resolution encounterd :{img.size}")
    return cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)  # back 2 cv2


def resize_and_save(
    fpath: Path,
    raw_data_dir: Path,
    out_data_dir: Path,
    resize_resolution: int,
    dataset_name: str,
):
    img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Error: {fpath} does not exist!")

    if dataset_name == "rfmid":
        img = crop_rfmid_img(img)
    img_resized = cv2.resize(
        img,
        (resize_resolution, resize_resolution),
        interpolation=cv2.INTER_AREA,
    )

    relative_path = fpath.relative_to(raw_data_dir)
    output_img_path = out_data_dir / relative_path.parent
    output_img_path.mkdir(parents=True, exist_ok=True)
    output_file = output_img_path / f"{fpath.stem}.png"
    cv2.imwrite(str(output_file), img_resized)  # type: ignore


parser = argparse.ArgumentParser(
    description="Arguments to preprocess raw images"
)
parser.add_argument(
    "--dataset", type=str, required=True, help="Dataset to preprocess"
)
parser.add_argument(
    "--raw-data-dir",
    type=str,
    required=True,
    help="Path to downloaded & unzipped dataset",
)
parser.add_argument(
    "--out-data-dir", type=str, required=True, help="Path to processed dataset"
)
parser.add_argument(
    "--resize",
    type=int,
    default=224,
    help="resize the image to which resolution",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=8,
    help="Assign workers to parallelize preprocessing",
)

if __name__ == "__main__":
    args = parser.parse_args()

    raw_data_dir, out_data_dir = Path(args.raw_data_dir), Path(
        args.out_data_dir
    )
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Error: The {raw_data_dir} does not exist!")

    if args.dataset == "isic2019":
        exts = ".jpg"
    elif args.dataset == "rfmid":
        exts = ".png"
    elif args.dataset == "nctcrc":
        exts = ".tif"
    elif args.dataset == "nearood":
        exts = [".png", ".jpg", ".jpeg"]
    else:
        raise NotImplementedError(
            f"{args.dataset} dataset is not yet supported"
        )

    img_paths_list = [
        file for file in raw_data_dir.rglob("*") if file.suffix.lower() in exts
    ]

    resize_func = partial(
        resize_and_save,
        raw_data_dir=raw_data_dir,
        out_data_dir=out_data_dir,
        resize_resolution=args.resize,
        dataset_name=args.dataset,
    )

    with Pool(args.num_workers) as pool:
        list(
            tqdm(
                pool.imap(resize_func, img_paths_list),
                total=len(img_paths_list),
            )
        )

    if args.dataset != "nearood":
        # no need to copy csv labels for ood-datasets
        for file in raw_data_dir.rglob("*.csv"):
            shutil.copy2(file, out_data_dir / file.name)

    # create val split for specific datasets
    if args.dataset == "isic2019":
        print(f"Creating validation split from train split")
        df = pd.read_csv(
            str(out_data_dir / "ISIC_2019_Training_GroundTruth.csv")
        )
        df["diagnosis"] = np.argmax(df.iloc[:, 1:].to_numpy(), axis=1)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df, val_df = train_df.drop("diagnosis", axis=1), val_df.drop(
            "diagnosis", axis=1
        )
        train_df.to_csv(
            str(out_data_dir / "ISIC_2019_Training_GroundTruth_split.csv"),
            index=False,
        )
        val_df.to_csv(
            str(out_data_dir / "ISIC_2019_Val_GroundTruth_split.csv"),
            index=False,
        )

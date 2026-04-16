import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import configurations
from pathlib import Path
import albumentations as A

from hydra import initialize, compose
from utils.make import make_datasets
from utils.logger import setup_logging

@hydra.main(config_name="expconf", version_base="1.3")
def main(configs: ExpConf):
    dotenv.load_dotenv(".env")
    configs.rootpath = Path(os.environ["ROOTPATH"])

    logdir = configs.rootpath / 'preprocessing'
    logger = setup_logging(log_dir=logdir)

    datasets = make_datasets(configs=configs, logger=logger)
    replay_transform = A.ReplayCompose([
        A.OneOf([
            # Noise-based corruptions
            A.GaussNoise(std_range=(0.2, 0.3), p=1.0),
            A.ISONoise(color_shift=(0.1, 0.5), intensity=(1.0, 2.0), p=1.0),

            # Blur & distortion corruptions
            A.MotionBlur(blur_limit=121, p=1),
            A.ZoomBlur(max_factor=5, p=1.0),

            # Weather-lik corruptions
            A.RandomSunFlare(src_radius=500, p=1.0),

            # Digital/artifact corruptions
            A.ImageCompression(quality_range=(1,10), p=1.0),
            A.Downscale(scale_range=(0.02, 0.10), p=1.0),
            A.PixelDropout(dropout_prob=0.5, p=1.0),
            A.GridDropout(ratio=0.75, p=1.0),
        ], p=1.0)
    ])

    pkl_fpath = (
        configs.rootpath /
        f'data/preprocessed/nearood/{configs.dataset.name}/corr/corr'
    )
    if not pkl_fpath.exists():
        pkl_fpath.mkdir(parents=True)
    pkl_fname = pkl_fpath / f'replay_{configs.dataset.name}.pkl'
    save_path = pkl_fpath

    # if you want to generate a replay
    ###################################
    # replay_list = []
    # aug_imgs = []

    # for i in tqdm(range(len(datasets["test"]))):
    #     batch = datasets["test"][i]
    #     image = np.transpose(batch["img"].numpy(), (1,2,0))
    #     result = replay_transform(image=image)

    #     aug_imgs.append(result["image"])
    #     replay_list.append(result["replay"])

    # with open(pkl_fname, "wb") as f:
    #     pickle.dump(replay_list, f)
    ###################################

    with open(pkl_fname, 'rb') as f:
        replay_list = pickle.load(f)

    for i in tqdm(range(len(datasets['test']))):
        image = datasets["test"][i]["img"].numpy()
        image = np.transpose(image, (1, 2, 0))
        result = A.ReplayCompose.replay(replay_list[i], image=image)
        image = result["image"]
        fname = f"{datasets['test'].labels_df.iloc[i]['Image Index']}.png"

        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        file_path = save_path / f"{fname}"
        pil_image.save(file_path)

if __name__ == "__main__":
    main()

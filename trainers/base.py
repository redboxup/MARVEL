import gc
import torch
import numpy as np
import tqdm.auto as tqdm
from logging import Logger
from types import GeneratorType
from contextlib import nullcontext
from torch.utils.data import DataLoader
from typing import Dict, Mapping, Tuple, Any
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from kornia.augmentation import AugmentationSequential

from utils import AverageMeter
from configurations import ExpConf
from datasets.base import BaseDataset
from utils.shuffle import GlobalBatchShuffler, shuffle_tensor_dicts_inplace

TRAINERS_REGISTRY = {}


class BaseTrainer:
    registered_trainers: dict[str, type] = TRAINERS_REGISTRY
    best_val_acc: float = 0
    best_val_dict: Dict[str, float]

    def __init__(
        self,
        configs: ExpConf,
        model: torch.nn.Module,
        dataloaders: dict[str, DataLoader[BaseDataset]],
        transforms: dict[str, AugmentationSequential],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        logger: Logger,
    ):

        self.configs = configs
        self.logger = logger

        self.device = configs.device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, Exiting...")

        self.model = model.to(self.device)
        self.dataloaders = dataloaders
        self.transforms = transforms
        self.losses = self.get_losses()

        if configs.trainer.prec == "amp":
            self.logger.info("Using Automatic Mixed Precision Training")
            self.model.float()  # convert model weights to fp32
            self.scaler = GradScaler(
                device=self.device
            )  # helps scale gradients dynamically

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.start_epoch = 0
        self.verbose(configs.verbose)

    def post_init(self, **kwargs):
        pass

    def get_losses(self) -> Mapping[str, torch.nn.Module]:
        raise NotImplementedError()

    def verbose(self, verbose=True):
        self.print = print if verbose else self.noop
        self.tqdm = tqdm.tqdm if verbose else self.noop_iter

    @staticmethod
    def noop(*args, **kwargs):
        pass

    @staticmethod
    def noop_iter(iterable, **kwargs):
        yield from iterable

    def train(self):
        self.logger.info("Training Started...")
        self.loss_weights = {loss: 1.0 for loss in self.losses}
        self.loss_weights.update(
            {
                name: param
                for name, param in self.configs.trainer.items()  # type: ignore
                if name in self.losses
            }
        )

        if self.configs.trainer.move_data_to_host:
            self.logger.info("Moving batches to host memory...")
            cache_dir = (
                self.configs.rootpath
                / 'data'
                / 'cache'
                / self.configs.dataset.name
            )
            use_cached_dataset = self.configs.trainer.cache_dataset
            using_aux_dataset = isinstance(
                self.iter_dataloaders(), GeneratorType
            )
            pt_name = 'train_aux.pt' if using_aux_dataset else 'train.pt'
            is_cache_available = (
                (cache_dir / pt_name).exists() and
                (cache_dir / 'val.pt').exists()
            )
            create_cache = use_cached_dataset and not is_cache_available

            if use_cached_dataset and is_cache_available:
                self.logger.info('Loading cache to host-memory')
                self.batches = torch.load(
                    cache_dir / pt_name, weights_only=False
                )
                self.val_batches = torch.load(
                    cache_dir / 'val.pt', weights_only=False,
                )
                # self.test_batches = torch.load(
                #     cache_dir / 'test.pt', weights_only=False,
                # )
                # set correct batch size
                self.logger.info('Setting batch-size')
                self.batches = GlobalBatchShuffler.batchify(
                    self.batches,
                    self.configs.trainer.batch_size,
                    multiple_loaders=True if using_aux_dataset else False,
                )
                self.val_batches = GlobalBatchShuffler.batchify(
                    self.val_batches,
                    self.configs.eval.batch_size,
                    multiple_loaders=False,
                )

            else:
                self.batches = list(
                    self.tqdm(
                        self.iter_dataloaders(),
                        total=len(self.dataloaders["train"]),
                        desc="Moving train split to host memory",
                    )
                )
                self.val_batches = list(
                    self.tqdm(
                        self.dataloaders["val"],
                        desc="Moving val split to host memory",
                    )
                )
                # self.test_batches = list(
                #     self.tqdm(
                #         self.dataloaders['test'],
                #         desc='Moving test split to host memory',
                #     )
                # )
                if create_cache:
                    self.logger.info('Saving a cache of the dataset')
                    cache_dir.mkdir(exist_ok=True, parents=True)
                    torch.save(
                        self.batches,
                        str(cache_dir / pt_name),
                        pickle_protocol=4,
                    )
                    torch.save(
                        self.val_batches,
                        str(cache_dir / 'val.pt'),
                        pickle_protocol=4,
                    )
                    # torch.save(
                    #     self.test_batches,
                    #     str(cache_dir / 'test.pt'),
                    #     pickle_protocol=4,
                    # )
            drop_last = (
                True
                if self.configs.trainer.name == 'marvel'
                else False
            )
            if drop_last:
                self.batches.pop()

        else:
            self.logger.info("Using lazy loading dataloader")
            self.batches = self.iter_dataloaders()
            self.val_batches = None

        # train loop
        for epoch in range(self.start_epoch, self.configs.trainer.epochs):
            self.model.train()
            self.epoch = epoch

            # train epoch
            self.logger.info(
                f"Epoch: {epoch} "
                f"lr: {self.optimizer.param_groups[0]['lr']:.4e}"
            )
            train_losses, extra_data = self.train_epoch()

            if self.scheduler:
                self.scheduler.step()

            is_last_epoch = (epoch == self.configs.trainer.epochs - 1)

            save_condition = is_last_epoch
            # save_condition = (
            #     epoch % self.configs.trainer.save_interval == 0
            # ) or ("best_val_acc" in extra_data) or (is_last_epoch)

            if save_condition:
                # best_val_acc = extra_data.get("best_val_acc", False)
                best_val_acc = False
                suffix = extra_data.get("suffix", "")

                self.save_checkpoint(epoch, best=best_val_acc, suffix=suffix)

    def iter_dataloaders(self):
        raise NotImplementedError()

    def model_forward(self, batch_input):
        raise NotImplementedError()

    def train_epoch(self):
        avg_losses = {
            loss_name: AverageMeter() for loss_name in self.losses
        }  # log losses

        if self.configs.trainer.move_data_to_host:
            using_aux_dataset = isinstance(
                self.iter_dataloaders(), GeneratorType
            )
            if using_aux_dataset:
                list_id, list_aux = map(list, zip(*self.batches))
                list_id = shuffle_tensor_dicts_inplace(list_id)
                list_aux = shuffle_tensor_dicts_inplace(list_aux)
                self.batches = list(zip(list_id, list_aux))
            else:
                self.batches = shuffle_tensor_dicts_inplace(
                    self.batches  # type: ignore
                )

        # use an autocontext if precision is 'amp'
        context = (
            autocast(self.device)
            if self.configs.trainer.prec == "amp"
            else nullcontext()
        )

        with context:
            # iterate over dataloader
            for batch_idx, batch_input in enumerate(
                self.tqdm(
                    (
                        self.batches
                        if self.configs.trainer.move_data_to_host
                        else self.iter_dataloaders()
                    ),
                    desc=f"Epoch: {self.epoch}",
                    total=len(self.dataloaders["train"]),
                )
            ):

                _input = batch_input
                try:
                    batch_size = _input["img"].shape[0]
                except:
                    batch_size = _input[0]["img"].shape[0]
                if batch_size == 0:
                    continue

                # forward pass
                outputs = self.model_forward(batch_input)

                loss_values = {
                    loss_name: self.compute_loss(
                        loss_name, loss, batch_input, outputs
                    )
                    for loss_name, loss in self.losses.items()
                }

                loss = torch.stack(
                    [
                        self.loss_weights[loss] * loss_values[loss]
                        for loss in loss_values
                    ]
                ).sum()

                # Backpropagation
                self.optimizer.zero_grad()
                (
                    self.scaler.scale(loss)
                    if self.configs.trainer.prec == "amp"
                    else loss
                ).backward()
                (
                    (self.scaler.step(self.optimizer), self.scaler.update())
                    if self.configs.trainer.prec == "amp"
                    else self.optimizer.step()
                )

                # Update averages
                for loss_name, loss_value in loss_values.items():
                    avg_losses[loss_name].update(
                        loss_value.item(), batch_size
                    )

                # In case of OOMs
                # if batch_idx % 10 == 0:
                #     gc.collect()
                #     torch.cuda.empty_cache()
                #     torch.cuda.synchronize()

        avg_losses = {
            loss_name: loss.avg for loss_name, loss in avg_losses.items()
        }
        # log losses
        msg = " ".join(
            [
                f"{loss_name} = {self.losses[loss_name].__class__.__name__}"
                f":{loss:.4f}"
                for loss_name, loss in avg_losses.items()
            ]
        )
        self.logger.info(msg)
        # extra data
        extra_data = self.post_train_epoch()
        return avg_losses, extra_data

    def post_train_epoch(self) -> dict[str, Any]:
        return {}

    def compute_loss(
        self, loss_name, loss_fn, batch_input, outputs
    ) -> torch.Tensor:
        raise NotImplementedError()

    def post_load_checkpoint(self, checkpoint_data):
        pass

    def save_checkpoint(self, epoch, best=False, suffix="", **extra_kwargs):
        if self.scheduler:
            extra_kwargs["scheduler"] = self.scheduler.state_dict()
        checkpoint_data = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            **extra_kwargs,
        }
        save_dir = self.configs.experiment_dir / "weights"
        if best:
            torch.save(checkpoint_data, save_dir / f"best_epoch{suffix}.pth")
            self.logger.info(
                f"Saving model — new best val metric: "
                f"{self.best_val_acc:.4f}"
            )
        else:
            torch.save(
                checkpoint_data, save_dir / f"epoch_{epoch}{suffix}.pth"
            )
            self.logger.info(
                f"Saving model at epoch {self.epoch} (interval save)"
            )

    def evaluate(self, split: str):
        return {}

    def load_checkpoint(
        self, epoch: int = 0, best: bool = False, suffix: str = ""
    ) -> None:
        state_dict_path = (
            self.configs.experiment_dir
            / "weights"
            / ("best_epoch.pth" if best else f"epoch_{epoch}{suffix}.pth")
        )

        state_dict: Dict[str, Any] = torch.load(
            state_dict_path, weights_only=False
        )

        for key, state in state_dict.items():
            module = getattr(self, key, None)

            if hasattr(module, "load_state_dict"):
                module.load_state_dict(state)  # type: ignore
                self.logger.info(
                    f"[green]{key} module has been loaded "
                    f"successfully![/green]"
                )
            else:
                setattr(self, key, state)
                self.logger.info(
                    f"{key} module has been set as trainer attribute"
                )

        self.logger.info(
            "[green]Loaded state dictionary successfully![/green]"
        )

    def eval_ood(self):
        return {}

    def train_stage_two(self):
        pass

    def forward_pass(
        self, name: str, desc: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs a forward pass through the model on the specified
        dataloader.

        This method is primarily used for evaluation or inference
        purposes. It does not perform any training or gradient updates.

        Args:
            name (str): The dataloader to use for the forward pass.
            desc (str): A custom description string to display in the
                        tqdm progress.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - logits (np.ndarray): Model outputs.
                - labels (np.ndarray): Corresponding ground truth labels.
        """
        logits_list, labels_list = [], []
        self.model.eval()
        with torch.inference_mode():
            for batch_input in self.tqdm(
                self.dataloaders[name],
                total=len(self.dataloaders[name]),
                desc=desc,
            ):
                images = self.transforms["test"](
                    batch_input["img"].to(self.device)
                )
                try:
                    labels = batch_input["label"]
                except KeyError:
                    labels = torch.zeros(len(batch_input["img"]))
                logits = self.model(images)
                logits_list.extend(logits.cpu().numpy())
                labels_list.extend(labels)

        logits_np = np.array(logits_list)
        labels_np = np.array(labels_list)
        return logits_np, labels_np

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        name = cls.__name__
        if not name.endswith("Trainer"):
            raise ValueError(
                f"Class '{name}' must end with 'Trainer' to be registered."
            )

        key = name.lower()[:-7]  # strip the 'Trainer'
        if key in cls.registered_trainers:
            raise KeyError(
                f"Duplicate trainer key '{key}' for class '{name}'."
            )

        cls.registered_trainers[key] = cls

    # def create_tsne(self):
    #     """
    #     Create t-SNE plots using features from id & ood datasets
    #     """
    #     import matplotlib.pyplot as plt
    #     from sklearn.manifold import TSNE

    #     splits = ["test", "nov_path", "nearood1", "nearood2",
    #               "nearood3", "farood"
    #               ]
    #     features = []
    #     flags = []

    #     self.model.eval()
    #     with torch.inference_mode():
    #         for split in splits:
    #             for batch in self.tqdm(
    #                 self.dataloaders[split],
    #                 desc=f"Extracting features: {split}"
    #             ):
    #                 images = batch['img'].to(self.device)
    #                 images = self.transforms['test'](images)
    #                 feats = self.model.forward_features(images)  # type: ignore
    #                 features.append(feats.cpu())
    #                 flags.extend([split] * feats.shape[0])
    #     # stack
    #     features = torch.cat(features, dim=0).numpy()
    #     tsne = TSNE(
    #         n_components=2,
    #         perplexity=30,
    #         learning_rate="auto",
    #         random_state=42,
    #         init="random",
    #         n_iter=1500,
    #     )
    #     tsne_out = tsne.fit_transform(features)

    #     # plot
    #     plt.figure(figsize=(8,8))
    #     unique_splits = sorted(set(flags))
    #     colors = {
    #         'test': 'blue',
    #         'nearood1': 'red',
    #         'nearood2': 'green',
    #         'nearood3': 'orange',
    #     }

    #     for s in unique_splits:
    #         idx = [i for i, flag in enumerate(flags) if flag == s]
    #         plt.scatter(
    #             tsne_out[idx, 0],
    #             tsne_out[idx, 1],
    #             s=8,
    #             alpha=0.7,
    #             label=s,
    #             c=colors.get(s, None)
    #         )
    #     plt.legend()
    #     plt.title('t-SNE of ID vs OOD Features')
    #     plt.tight_layout()

    #     plt.savefig('tsne.png', dpi=300)

    def create_tsne(self):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE  # Use sklearn instead of umap-learn
        import torch
        import numpy as np

        features = []
        labels = []

        self.model.eval()
        split = "test"

        with torch.inference_mode():
            for batch in self.tqdm(self.dataloaders[split], desc="Extracting features"):
                images = batch['img'].to(self.device)
                images = self.transforms['test'](images)
                
                feats = self.model.forward_features(images)
                features.append(feats.cpu())
                
                # Extract labels for class-based coloring
                batch_labels = batch['label']
                labels.extend(batch_labels.tolist())

        features = torch.cat(features, dim=0).numpy()
        # Minimal t-SNE setup
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            random_state=42,
            init="pca",       # "pca" initialization is often more stable/faster
            learning_rate="auto"
        )
        tsne_out = tsne.fit_transform(features)

        # Plotting
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(
            tsne_out[:, 0], 
            tsne_out[:, 1], 
            c=labels, 
            cmap='tab10',
            s=20, 
            alpha=0.7
        )
        unique_labels = np.unique(labels)
        legend1 = plt.legend(*scatter.legend_elements(num=len(unique_labels)), title="Classes", loc="upper right")
        plt.gca().add_artist(legend1)
        plt.show()
        plt.savefig('tsne.png')

    def create_umap(self):
        """
        Create UMAP plots using features from the test dataset, 
        colored by class labels.
        """
        import umap
        import matplotlib.pyplot as plt

        features = []
        labels = []

        self.model.eval()
        # We only care about the 'test' split for this visualization
        split = "test"
        
        if split not in self.dataloaders:
            print(f"Error: {split} dataloader not found.")
            return

        with torch.inference_mode():
            for batch in self.tqdm(
                self.dataloaders[split], 
                desc=f"Extracting UMAP features: {split}"
            ):
                images = batch['img'].to(self.device)
                # Apply test transforms if applicable
                if 'test' in self.transforms:
                    images = self.transforms['test'](images)
                
                # Extract features
                feats = self.model.forward_features(images)
                features.append(feats.cpu())
                
                # Collect labels for coloring
                # Assuming labels are stored under 'label' or 'target' in your batch
                batch_labels = batch['label']
                if batch_labels is not None:
                    labels.extend(batch_labels.tolist())

        # Prepare data for UMAP
        features = torch.cat(features, dim=0).numpy()
        labels = np.array(labels)

        # Initialize and fit UMAP
        # n_neighbors: balances local vs global structure (similar to perplexity)
        # min_dist: controls how tightly points are packed together
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=42
        )
        
        umap_out = reducer.fit_transform(features)

        # Plotting
        plt.figure(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        # Using a qualitative colormap for distinct classes
        scatter = plt.scatter(
            umap_out[:, 0], 
            umap_out[:, 1], 
            c=labels, 
            cmap='Spectral', 
            s=10, 
            alpha=0.8
        )
        
        # Create a legend with class names if available, otherwise use IDs
        plt.colorbar(scatter, label='Class ID')
        plt.title(f'UMAP Projection of {split.upper()} Features by Class')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.show()

# Registry for trainers
BaseTrainer.registered_trainers = {}

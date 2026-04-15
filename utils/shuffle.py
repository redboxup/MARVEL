import torch
from typing import Tuple, List, Any, Dict

from utils.reproducibility import deterministic_randperm


class GlobalBatchShuffler:
    @staticmethod
    def _detect_structure(batches: List[Any]) -> Tuple[List[int], str]:
        if isinstance(batches[0], torch.Tensor):
            batch_sizes = [batch.size(0) for batch in batches]
            return batch_sizes, "tensor"

        elif isinstance(batches[0], (tuple, list)):
            batch_sizes = [batch[0].size(0) for batch in batches]
            return batch_sizes, "tuple"

        elif isinstance(batches[0], dict):
            batch_sizes = [
                next(iter(batch.values())).size(0) for batch in batches
            ]
            return batch_sizes, "dict"
        else:
            raise TypeError(f"Unsupported batch format: {type(batches[0])}")

    @staticmethod
    def _flatten_batches(
        batches: List[Any], data_format: str
    ) -> Tuple[Any, int]:
        if data_format == "tensor":
            flat_data = torch.cat(batches, dim=0)
            return flat_data, flat_data.size(0)

        elif data_format == "tuple":
            flat_data = tuple(
                torch.cat(parts, dim=0) for parts in zip(*batches)
            )
            return flat_data, flat_data[0].size(0)

        elif data_format == "dict":
            keys = batches[0].keys()
            flat_data = {
                key: torch.cat([batch[key] for batch in batches], dim=0)
                for key in keys
            }
            return flat_data, next(iter(flat_data.values())).size(0)
        else:
            raise TypeError(f"Unsupported data format: {data_format}")

    @staticmethod
    def _shuffle(
        flat_data: Any, data_format: str, total: int, seed: int
    ) -> Any:
        perm = deterministic_randperm(length=total, seed=seed)

        if data_format == "tensor":
            return flat_data[perm]

        elif data_format == "tuple":
            return tuple(part[perm] for part in flat_data)

        elif data_format == "dict":
            return {key: value[perm] for key, value in flat_data.items()}

    @staticmethod
    def _create_batches(flat_data, data_format: str, batch_sizes: List[int]):
        indices = [0] + list(
            torch.cumsum(torch.tensor(batch_sizes), dim=0).tolist()
        )

        if data_format == "tensor":
            return [
                flat_data[indices[i]: indices[i + 1]]
                for i in range(len(batch_sizes))
            ]

        elif data_format == "tuple":
            return [
                tuple(part[indices[i]: indices[i + 1]] for part in flat_data)
                for i in range(len(batch_sizes))
            ]

        elif data_format == "dict":
            return [
                {
                    key: flat_data[key][indices[i]: indices[i + 1]]
                    for key in flat_data
                }
                for i in range(len(batch_sizes))
            ]

    @staticmethod
    def shuffle_batches(
        batches: Any, seed: int = 42, multiple_loaders: bool = False
    ) -> Any:

        # if list is empty
        if not batches:
            return []

        # if iterating over multiple dataloaders
        if multiple_loaders:
            shuffled_batches = []
            list_batches = [item for item in zip(*batches)]
            for i, batches in enumerate(list_batches):
                batch_sizes, data_format = (
                    GlobalBatchShuffler._detect_structure(batches)
                )  # detect structure
                flat_data, total = GlobalBatchShuffler._flatten_batches(
                    batches, data_format
                )  # flattend the data
                flat_data = GlobalBatchShuffler._shuffle(
                    flat_data, data_format, total, seed=seed + i
                )  # Shuffle
                new_batches = GlobalBatchShuffler._create_batches(
                    flat_data, data_format, batch_sizes
                )  # batchify
                shuffled_batches.append(new_batches)
            return [batches for batches in zip(*shuffled_batches)]

        # iterating over single dataloader
        batch_sizes, data_format = GlobalBatchShuffler._detect_structure(
            batches
        )  # detect structure
        flat_data, total = GlobalBatchShuffler._flatten_batches(
            batches, data_format
        )  # flattend the data
        flat_data = GlobalBatchShuffler._shuffle(
            flat_data, data_format, total, seed=seed
        )  # Shuffle
        new_batches = GlobalBatchShuffler._create_batches(
            flat_data, data_format, batch_sizes
        )  # batchify

        return new_batches

    @staticmethod
    def batchify(
        batches: Any, batch_size: int, multiple_loaders: bool = True,
    ):
        def _batchify(batches: Any):
            _, data_format = GlobalBatchShuffler._detect_structure(
                batches
            )
            flat_data, total_ = GlobalBatchShuffler._flatten_batches(
                batches, data_format,
            )
            batch_sizes = list(range(0, total_, batch_size))[1:] + [total_]
            batch_sizes = (
                [batch_size] * (total_ // batch_size) +
                ([total_ % batch_size] if total_ % batch_size != 0 else [])
            )
            batches = GlobalBatchShuffler._create_batches(
                flat_data, data_format, batch_sizes,
            )
            return batches

        if multiple_loaders:
            list_new_batches = [_batchify(item) for item in zip(*batches)]
            return list(zip(*list_new_batches))

        else:
            return _batchify(batches)


def shuffle_tensor_dicts_inplace(dict_list: List[Dict[str, torch.Tensor]]):
    if not dict_list:
        return dict_list

    batch_sizes = [list(d.values())[0].shape[0] for d in dict_list]
    total_images = sum(batch_sizes)

    cumulative_sizes = [0]
    for size in batch_sizes:
        cumulative_sizes.append(cumulative_sizes[-1] + size)

    def global_to_local(global_idx):
        for i in range(len(cumulative_sizes) - 1):
            if global_idx < cumulative_sizes[i + 1]:
                return i, global_idx - cumulative_sizes[i]
        raise ValueError(
            f"Global index {global_idx} is out of bounds"
            f" (max: {total_images - 1})"
        )

    # Fisher-Yates shuffle
    for i in range(total_images - 1, 0, -1):
        j = int(torch.randint(0, i + 1, (1,)).item())

        if i != j:
            dict_i, local_i = global_to_local(i)
            dict_j, local_j = global_to_local(j)

            for key in dict_list[0].keys():
                temp = dict_list[dict_i][key][local_i].clone()
                dict_list[dict_i][key][local_i] = (
                    dict_list[dict_j][key][local_j]
                )
                dict_list[dict_j][key][local_j] = temp

    return dict_list

from rich.console import Console
from rich.table import Table
from configurations import ExpConf
import pandas as pd


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0  # Current value
        self.avg = 0  # Average value
        self.sum = 0  # Sum of values
        self.count = 0  # Number of values

    def update(self, val, n=1):
        """Update the meter with a new value.

        Args:
            val (float): The new value to add.
            n (int): The number of occurrences of this value (default is 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self):
        return f"Current: {self.val:.4f}, Average: {self.avg:.4f}"


def console_plot_tables(configs: ExpConf):
    """Call at the end, print pretty tables using csv files saved in results_directory

    Args:
        configs (ExpConf): hydra configs file
    """
    console = Console()

    # read results directory and load all csv files
    csv_files = [
        csv_file for csv_file in configs.trainer.results_dir.rglob("*.csv")
    ]
    csv_files = sorted(csv_files)

    # iterate over all csv files
    for csv_file in csv_files:
        if csv_file.name == "ID_test_auc.csv":
            df = pd.read_csv(csv_file)
            table = Table(
                title=f"{configs.dataset.datasetname} Classification Results",
                show_header=True,
                header_style="bold magenta",
            )
            # Add columns
            table.add_column(
                "Pathology", justify="left", style="cyan", no_wrap=True
            )
            table.add_column("Score", justify="center", style="green")

            # Add rows
            for condition, auc_score in df.items():
                table.add_row(f"{condition[:-4]}", f"{float(auc_score):.4f}")
            console.print(table)

        elif csv_file.name == "ID_ConfInterval_auc.csv":
            df = pd.read_csv(csv_file)
            table = Table(
                title="CXR Classification AUC Scores with 95% CI",
                show_header=True,
                header_style="bold magenta",
            )
            # Add columns
            table.add_column(
                "Condition", justify="left", style="cyan", no_wrap=True
            )
            table.add_column(
                "AUC Score (95% CI)", justify="center", style="green"
            )

            # Add rows
            for condition, condition_data in df.items():
                # Format as "Mean [Lower, Upper]"
                ci_text = f"{condition_data.iloc[0]} [{condition_data.iloc[1]}, {condition_data.iloc[2]}]"
                table.add_row(condition[:-4], ci_text)
            console.print(table)

        elif f"OOD_{configs.eval.name}" in csv_file.name:
            df = pd.read_csv(csv_file)
            table = Table(
                title=f"OOD Detection in {csv_file.name[4:-4]} ",
                show_header=True,
                header_style="bold magenta",
            )
            # Add columns
            table.add_column(
                "metrics", justify="left", style="cyan", no_wrap=True
            )
            table.add_column("score", justify="center", style="green")

            # Add rows
            for metric, metrics_data in df.items():
                val = f"{metrics_data.iloc[0]:.4f}"
                table.add_row(metric, val)
            console.print(table)

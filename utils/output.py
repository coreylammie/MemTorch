import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml
from codetiming import Timer
from torch.nn import Module

from utils.logger import logger
from utils.settings import settings

OUT_DIR = "./out"
OUT_FILES = {
    "settings": "settings.yaml",
    "results": "results.yaml",
    "network_info": "network_info.yaml",
    "timers": "timers.yaml",
}


def init_out_directory() -> None:
    """
    Prepare the output directory.
    """

    # Skip saving if the name of the run is not set
    if not settings.run_name:
        logger.warning(
            "Nothing will be saved because the name of the run is not set. "
            'See "run_name" in the setting file to change this behaviours.'
        )
        return

    run_dir = Path(OUT_DIR, settings.run_name)
    img_dir = run_dir / "img"

    # If the keyword 'tmp' is used as run name, then remove the previous files
    if settings.run_name == "tmp":
        logger.warning(f"Using temporary directory to save this run results.")
        if run_dir.exists():
            logger.warning(f"Previous temporary files removed: {run_dir}")
            # Remove text files
            (run_dir / "run.log").unlink(missing_ok=True)
            for file_name in OUT_FILES.values():
                (run_dir / file_name).unlink(missing_ok=True)

            # Remove images
            if img_dir.is_dir():
                # Remove png images files
                for png_file in img_dir.glob("*.png"):
                    png_file.unlink()
                img_dir.rmdir()

            # Remove saved networks
            for p_file in run_dir.glob("*.p"):
                p_file.unlink()

            # Remove tmp directory
            run_dir.rmdir()

    try:
        # Create the directories
        img_dir.mkdir(parents=True)
    except FileExistsError as err:
        # Clear error message about file exist
        raise RuntimeError(
            f'The run name "{settings.run_name}" is already used '
            f'in the out directory "{run_dir}". '
            f'Change the name in the run settings to a new one or "tmp" or empty.'
        ) from err

    logger.debug(f"Output directory created: {run_dir}")

    # Init the logger file
    if settings.logger_file_enable:
        logger.enable_log_file(
            file_path=(run_dir / "run.log"), file_log_level=settings.logger_file_level
        )

    parameter_file = run_dir / OUT_FILES["settings"]
    with open(parameter_file, "w+") as f:
        yaml.dump(asdict(settings), f)

    logger.debug(f"Parameters saved in {parameter_file}")


def set_plot_style():
    """
    Set plot style.
    """
    sns.set_theme(
        rc={"axes.titlesize": 15, "axes.labelsize": 13, "figure.autolayout": True}
    )


def save_network_info(network_metrics: dict) -> None:
    """
    Save metrics information in a file in the run directory.

    :param network_metrics: The dictionary of metrics with their values.
    """

    # Skip saving if the name of the run is not set or nothing to save
    if not settings.run_name or len(network_metrics) == 0:
        return

    network_info_file = Path(OUT_DIR, settings.run_name, OUT_FILES["network_info"])

    with open(network_info_file, "w+") as f:
        yaml.dump(network_metrics, f)

    logger.debug(f"Network info saved in {network_info_file}")


def save_results(**results: Any) -> None:
    """
    Write a new line in the result file.

    :param results: Dictionary of labels and values, could be anything that implement __str__.
    """

    # Skip saving if the name of the run is not set
    if not settings.run_name:
        return

    results_path = Path(OUT_DIR, settings.run_name, OUT_FILES["results"])

    # Append to the file, create it if necessary
    with open(results_path, "a") as f:
        yaml.dump(results, f)

    logger.debug(f"{len(results)} result(s) saved in {results_path}")


def save_plot(file_name: str) -> None:
    """
    Save a plot image in the directory
    """

    # Skip saving if the name of the run is not set
    if not settings.run_name:
        return

    save_path = Path(OUT_DIR, settings.run_name, "img", f"{file_name}.png")

    plt.savefig(save_path)
    logger.debug(f"Plot saved in {save_path}")

    # Plot image or close it
    plt.show(block=False) if settings.show_images else plt.close()


def save_network(network: Module, file_name: str = "network") -> None:
    """
    Save a full description of the network parameters and states.

    :param network: The network to save
    :param file_name: The name of the destination file (without the extension)
    """

    # Skip saving if the name of the run is not set
    if not settings.run_name:
        return

    cache_path = Path(OUT_DIR, settings.run_name, file_name + ".p")
    torch.save(network.state_dict(), cache_path)
    logger.debug(f"Network saved in {cache_path}")


def save_timers() -> None:
    """
    Save the named timers in a file in the output directory.
    """

    # Skip saving if the name of the run is not set or nothing to save
    if not settings.run_name or len(Timer.timers.data) == 0:
        return

    timers_file = Path(OUT_DIR, settings.run_name, OUT_FILES["timers"])
    with open(timers_file, "w+") as f:
        # Save with replacing white spaces by '_' in timers name
        f.write("# Values in seconds\n")
        yaml.dump(
            {re.sub(r"\s+", "_", n.strip()): v for n, v in Timer.timers.data.items()}, f
        )

    logger.debug(f"{len(Timer.timers.data)} timer(s) saved in {timers_file}")


def load_network(network: Module, file_path: Union[str, Path]) -> bool:
    """
    Load a full description of the network parameters and states from a previous save file.

    :param network: The network to load into (in place)
    :param file_path: The path to the file to load
    :return: True if the file exist and is loaded, False if the file is not found.
    """

    cache_path = Path(file_path) if isinstance(file_path, str) else file_path
    if cache_path.is_file():
        network.load_state_dict(torch.load(cache_path))
        logger.info(f"Network loaded ({cache_path})")
        return True
    logger.warning(f'Network cache not found in "{cache_path}"')
    return False


def load_run_files(dir_path: Path) -> dict:
    """
    Load all the information of a run from its files.

    :param dir_path: The path to the directory of the run
    :return: A dictionary of every value starting with the name of the file ("file.key": value)
    """
    data = {}

    # For each output file of the run
    for key, file in OUT_FILES.items():
        with open(dir_path / file) as f:
            content = yaml.load(f, Loader=yaml.FullLoader)
            # For each value of each file
            for label, value in content.items():
                data[key + "." + label] = value

    return data


def load_runs(pattern: str) -> pd.DataFrame:
    """
    Load all informations form files in the out directory matching with a pattern.

    :param pattern: The pattern to filter runs
    :return: A dataframe containing all information, with the columns as "file.key"
    """
    data = []

    runs_dir = Path(OUT_DIR)
    for run_dir in runs_dir.glob(pattern):
        data.append(load_run_files(run_dir))

    logger.info(f'{len(data)} run(s) loaded with the pattern "{runs_dir}/{pattern}"')

    return pd.DataFrame(data)

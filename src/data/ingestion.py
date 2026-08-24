from pathlib import Path

import pandas as pd


def load_raw_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load the household electricity consumption dataset.

    Parameters
    ----------
    file_path : str | Path
        Path to the raw household power consumption file.

    Returns
    -------
    pd.DataFrame
        Raw electricity consumption data.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep=";",
        low_memory=False
    )

    return df
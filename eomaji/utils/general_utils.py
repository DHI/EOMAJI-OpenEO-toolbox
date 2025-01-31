import pandas as pd
import importlib.resources as pkg_resources
import eomaji.static_data


def load_lut():
    """Loads the WorldCover10m lookup table from the package data directory."""
    with (
        pkg_resources.files(eomaji.static_data)
        .joinpath("WorldCover10m_2020_LUT.csv")
        .open("r") as file
    ):
        lut = pd.read_csv(file, sep=";")
    return lut

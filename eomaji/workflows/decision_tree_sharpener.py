import os
import logging
import tempfile
import openeo
import xarray as xr
from pyDMS.pyDMS import DecisionTreeSharpener
from eomaji.utils import gdal_to_xarray


# Setting up logger
logger = logging.getLogger(__name__)


def run_decision_tree_sharpener(
    high_resolution_cube: openeo.DataCube,
    low_resolution_cube: openeo.DataCube,
) -> xr.DataArray:
    """
    Perform disaggregation of low-resolution imagery to high-resolution imagery using the
    DecisionTreeSharpener algorithm.

    Args:
        high_resolution_cube: openeo.DataCube,
        low_resolution_cube: openeo.DataCube,

    Returns:
        xarray.DataArray: Downscaled image as an xarray.
    """
    try:
        # Download the high-resolution file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as high_res_temp:
            high_res_file = high_res_temp.name
            high_resolution_cube.download(high_res_file)
            logger.info(f"Downloaded high-resolution file to {high_res_file}")

        # Download the low-resolution file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as low_res_temp:
            low_res_file = low_res_temp.name
            low_resolution_cube.download(low_res_file)
            logger.info(f"Downloaded low-resolution file to {low_res_file}")

        # Common options for the decision tree model
        common_opts = {
            "highResFiles": [high_res_file],  # Local path to high-res image
            "lowResFiles": [low_res_file],  # Local path to low-res image
            "lowResGoodQualityFlags": [1],
            "cvHomogeneityThreshold": 0,
            "movingWindowSize": 30,  # Adjust based on spatial scale
            "disaggregatingTemperature": True,
            "baggingRegressorOpt": {
                "n_jobs": 3,  # Number of parallel jobs
                "n_estimators": 30,  # Number of decision trees in the ensemble
                "max_samples": 0.8,  # Proportion of samples for each tree
                "max_features": 0.8,  # Proportion of features for each tree
            },
        }

        # Decision tree options for disaggregation
        dt_opts = common_opts.copy()
        dt_opts["perLeafLinearRegression"] = True
        dt_opts["linearRegressionExtrapolationRatio"] = 0.25

        # Initialize and train the disaggregator
        disaggregator = DecisionTreeSharpener(**dt_opts)
        disaggregator.trainSharpener()

        # Apply sharpener to the images
        downscaled_image = disaggregator.applySharpener(
            highResFilename=high_res_file, lowResFilename=low_res_file
        )

        # Convert the result to xarray format
        downscaled_image_xarray = gdal_to_xarray(downscaled_image)

        # Cleanup temporary files
        os.remove(high_res_file)
        os.remove(low_res_file)
        logger.info(f"Temporary files {high_res_file} and {low_res_file} removed.")

        return downscaled_image_xarray

    except Exception as e:
        logger.error(f"An error occurred during disaggregation: {e}")
        raise

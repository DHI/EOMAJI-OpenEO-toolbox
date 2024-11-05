# /// script
# dependencies = [
#   "pyDMS @ https://github.com/mariegitter/pyDMS/archive/refs/tags/v0.3.zip",
# ]
# ///
import numpy as np
import xarray as xr
from openeo.udf import XarrayDataCube
from openeo.udf import inspect
from pyDMS.pyDMS import DecisionTreeSharpener


def gdal_to_xarray(gdal_dataset):
    # Get the dimensions
    x_size = gdal_dataset.RasterXSize
    y_size = gdal_dataset.RasterYSize
    num_bands = gdal_dataset.RasterCount

    # Initialize an empty array to store the data
    data = np.zeros((num_bands, y_size, x_size), dtype=np.float32)

    # Loop through each band and read the data into the array
    for i in range(num_bands):
        band = gdal_dataset.GetRasterBand(i + 1)
        data[i, :, :] = band.ReadAsArray()

    # Get geotransform and projection information
    geotransform = gdal_dataset.GetGeoTransform()
    projection = gdal_dataset.GetProjection()

    # Define coordinates
    x_coords = geotransform[0] + np.arange(x_size) * geotransform[1]
    y_coords = geotransform[3] + np.arange(y_size) * geotransform[5]
    band_coords = np.arange(1, num_bands + 1)

    # Create an xarray Dataset
    ds = xr.Dataset(
        {"band_data": (["band", "y", "x"], data)},
        coords={"x": x_coords, "y": y_coords, "band": band_coords},
        attrs={"crs": projection, "transform": geotransform},
    )
    return ds


def apply_datacube(data: XarrayDataCube, context: dict) -> XarrayDataCube:
    """
    This function applies the Decision Tree Sharpener to the input data cube.

    Parameters:
    - input_datacube: None - dummy variable
    - context: A dictionary that should provide necessary context, including high-res image and mask.

    Returns:
    - Output XarrayDataCube with sharpened data.
    """
    inspect(data=[len(context)], message="UDF logging len of context")
    high_res_file = context["high_res_file"]
    low_res_file = context["low_res_file"]

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

    dt_opts = common_opts.copy()
    dt_opts["perLeafLinearRegression"] = True
    dt_opts["linearRegressionExtrapolationRatio"] = 0.25

    disaggregator = DecisionTreeSharpener(**dt_opts)
    disaggregator.trainSharpener()
    downscaled_image = disaggregator.applySharpener(
        highResFilename=high_res_file, lowResFilename=low_res_file
    )

    downscaled_image_xarray = gdal_to_xarray(downscaled_image)

    return downscaled_image_xarray

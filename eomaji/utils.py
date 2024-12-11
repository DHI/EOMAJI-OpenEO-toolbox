import numpy as np
import xarray as xr
from osgeo import gdal


def gdal_to_xarray(gdal_dataset: gdal.Dataset):
    """
    Convert a GDAL dataset to an xarray Dataset.

    This function extracts raster data (bands, geospatial coordinates, and projection information)
    from a GDAL dataset and returns it as an xarray Dataset. Each band in the GDAL dataset is
    represented as a 2D array, and the xarray Dataset includes coordinates for both spatial
    dimensions (x, y) and the bands.

    Args:
        gdal_dataset (gdal.Dataset): The GDAL dataset object containing raster data. This should be
                                      a loaded raster image or a spatial dataset (e.g., a GeoTIFF).

    Returns:
        xarray.Dataset: An xarray Dataset containing the raster data, with coordinates for x, y,
                        and band dimensions, and attributes for CRS (coordinate reference system)
                        and geotransform.
    """
    # Get the dimensions of the raster (x, y, number of bands)
    x_size = gdal_dataset.RasterXSize
    y_size = gdal_dataset.RasterYSize
    num_bands = gdal_dataset.RasterCount

    # Initialize an empty array to store the data (shape: [num_bands, y_size, x_size])
    data = np.zeros((num_bands, y_size, x_size), dtype=np.float32)

    # Loop through each band and read the data into the array
    for i in range(num_bands):
        band = gdal_dataset.GetRasterBand(i + 1)  # Bands are 1-indexed in GDAL
        data[i, :, :] = band.ReadAsArray()

    # Get geotransform and projection information
    geotransform = gdal_dataset.GetGeoTransform()
    projection = gdal_dataset.GetProjection()

    # Define the x and y coordinates based on the geotransform
    x_coords = (
        geotransform[0] + np.arange(x_size) * geotransform[1]
    )  # X coordinates (longitude)
    y_coords = (
        geotransform[3] + np.arange(y_size) * geotransform[5]
    )  # Y coordinates (latitude)
    band_coords = np.arange(1, num_bands + 1)  # Band coordinates (starting from 1)

    # Create an xarray Dataset with the raster data
    ds = xr.Dataset(
        {"band_data": (["band", "y", "x"], data)},  # Data variable with coordinates
        coords={
            "x": x_coords,  # X coordinates (e.g., longitude or easting)
            "y": y_coords,  # Y coordinates (e.g., latitude or northing)
            "band": band_coords,  # Band index coordinates
        },
        attrs={
            "crs": projection,  # Coordinate Reference System
            "transform": geotransform,  # Geotransform (affine transformation)
        },
    )

    return ds

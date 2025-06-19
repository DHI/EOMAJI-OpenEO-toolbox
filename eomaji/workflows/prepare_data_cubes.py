from typing import List, Tuple
import datetime
import os
from pathlib import Path
import openeo
import hashlib
import logging
from shapely.geometry import box
from shapely import to_geojson
from dateutil.relativedelta import relativedelta
from eomaji.workflows import sentinel2_preprocessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def prepare_data_cubes(
    connection: openeo.Connection,
    bbox: List | Tuple,
    date: datetime.date | datetime.datetime,
    sentinel2_search_range: int = 3,
    out_dir: str = "./data",
):
    """
    Prepare and cache Sentinel-2 and Sentinel-3 data cubes for a given AOI and date.

    This function retrieves and preprocesses remote sensing datasets from the OpenEO platform:
    - Sentinel-2: Optical bands and biophysical variables (e.g., LAI, FAPAR, FCOVER, CCC, CWC)
    - Sentinel-3: Land Surface Temperature (LST), confidence and viewing angle
    - WorldCover 2021: Land cover classification
    - Copernicus DEM: Digital elevation model for spatial resampling

    All data is masked, reduced, and resampled as appropriate, and stored locally as NetCDF or GeoTIFF files.
    Existing outputs are reused to avoid unnecessary recomputation.

    Args:
        connection (openeo.Connection): An active OpenEO connection.
        bbox (List[float] | Tuple[float, float, float, float]): Bounding box (west, south, east, north).
        date (datetime.date | datetime.datetime): Center date for data search and acquisition.
        sentinel2_search_range (int, optional): Number of days before and after `date` for to search for cloud free Sentinel-2. Defaults to 3.
        out_dir (str, optional): Directory to store the cached files. Defaults to "./data".

    Returns:
        Tuple[str, str, str, str, str, float]:
            - s2_path (str): Path to saved Sentinel-2 data cube (.nc)
            - s3_path (str): Path to saved Sentinel-3 data cube (.nc)
            - worldcover_path (str): Path to WorldCover 2021 GeoTIFF
            - dem_s2_path (str): Path to DEM resampled to Sentinel-2 resolution (.tif)
            - dem_s3_path (str): Path to DEM resampled to Sentinel-3 resolution (.tif)
            - acq_time (float): Sentinel-3 acquisition hour (UTC)
    """

    # Convert date to string for path name
    date_str = str(date).replace("-", "")

    # Generate a hash based on the bounding box coordinates
    bbox_hash = hashlib.md5(str(bbox).encode()).hexdigest()[
        :8
    ]  # Short hash for path name

    # Base directory includes date and bbox hash
    base_dir = os.path.join(out_dir, f"{date_str}_{bbox_hash}")
    os.makedirs(base_dir, exist_ok=True)

    # Define output file paths
    s2_path = os.path.join(base_dir, "s2_data.nc")
    s3_path = os.path.join(base_dir, "s3_data.nc")
    dem_s2_path = Path(base_dir) / f"{date_str}_ELEV.tif"
    dem_s3_path = Path(base_dir) / "meteo_dem.tif"
    worldcover_path = Path(base_dir) / "WordlCover2021.tif"

    # if (
    #    os.path.exists(s2_path)
    #    and os.path.exists(s3_path)
    #    and os.path.exists(dem_s2_path)
    #    and os.path.exists(worldcover_path)
    # ):
    #   logging.info("Cached data cubes found. Skipping download.")
    #    return s2_path, s3_path

    # Prepare AOI and date range
    aoi = dict(zip(["west", "south", "east", "north"], bbox))
    bbox_polygon = eval(to_geojson(box(*bbox)))
    time_window = [
        str(date + relativedelta(days=-sentinel2_search_range)),
        str(date + relativedelta(days=+sentinel2_search_range)),
    ]

    # Define bands
    s2_bands = [
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
        "SCL",
        "sunZenithAngles",
    ]

    # Load Biopar data
    fapar = sentinel2_preprocessing.get_biopar(
        connection, "FAPAR", time_window, bbox_polygon
    )
    lai = sentinel2_preprocessing.get_biopar(
        connection, "LAI", time_window, bbox_polygon
    )
    fcover = sentinel2_preprocessing.get_biopar(
        connection, "FCOVER", time_window, bbox_polygon
    )
    ccc = sentinel2_preprocessing.get_biopar(
        connection, "CCC", time_window, bbox_polygon
    )
    cwc = sentinel2_preprocessing.get_biopar(
        connection, "CWC", time_window, bbox_polygon
    )

    # Load Sentinel-2 cube and merge with Biopar
    s2_cube = connection.load_collection(
        "SENTINEL2_L2A", spatial_extent=aoi, temporal_extent=time_window, bands=s2_bands
    )
    merged = (
        fapar.merge_cubes(lai)
        .merge_cubes(fcover)
        .merge_cubes(ccc)
        .merge_cubes(cwc)
        .merge_cubes(s2_cube)
    )

    # Apply cloud and shadow mask using SCL (keep only class 4 and 5 = vegetation/bare)
    mask = ~((merged.band("SCL") == 4) | (merged.band("SCL") == 5))
    masked = merged.mask(mask)

    # Reduce time dimension by selecting the first valid observation
    s2_best_pixel = masked.reduce_dimension(dimension="t", reducer="first")

    if not os.path.exists(s2_path):
        s2_best_pixel.execute_batch(s2_path)
    else:
        logging.info("Cached Sentinel 2 data cube found. Skipping download.")

    # Load Sentinel-3 data
    s3_cube = connection.load_collection(
        "SENTINEL3_SLSTR_L2_LST",
        spatial_extent=aoi,
        temporal_extent=[str(date), str(date)],
        bands=["LST", "confidence_in", "viewZenithAngles"],
        properties={
            "timeliness": lambda x: x == "NT",
            "orbitDirection": lambda x: x == "DESCENDING",
        },
    )
    acq_time = float(
        s3_cube.metadata.temporal_dimension.extent[0].split("T")[1].split(":")[0]
    )
    if not os.path.exists(s3_path):
        s3_cube.execute_batch(s3_path)

    dem_cube = connection.load_collection("COPERNICUS_30", spatial_extent=aoi)
    dem_resampled_s2_cube = dem_cube.resample_cube_spatial(s2_cube, method="bilinear")
    dem_resampled_s3_cube = dem_cube.resample_cube_spatial(s3_cube, method="bilinear")
    if not os.path.exists(dem_s2_path):
        dem_resampled_s2_cube.execute_batch(dem_s2_path)
    else:
        logging.info("Cached DEM data cube found. Skipping download.")
    if not os.path.exists(dem_s3_path):
        dem_resampled_s3_cube.execute_batch(dem_s3_path)
    else:
        logging.info("Cached DEM cube found. Skipping download.")

    worldcover = connection.load_collection(
        "ESA_WORLDCOVER_10M_2021_V2", temporal_extent=["2021-01-01", "2021-12-31"]
    ).filter_bbox(bbox)
    wc_resampled_s2_cube = worldcover.resample_cube_spatial(s2_cube, method="near")
    if not os.path.exists(worldcover_path):
        wc_resampled_s2_cube.execute_batch(worldcover_path)
    else:
        logging.info("Cached Worldcover cube found. Skipping download.")

    # lst_resampled_cube = lst_cube.resample_cube_spatial(s2_full_cube, method="bilinear")
    logging.info("Data cubes prepared and saved.")

    return s2_path, s3_path, worldcover_path, dem_s2_path, dem_s3_path, acq_time

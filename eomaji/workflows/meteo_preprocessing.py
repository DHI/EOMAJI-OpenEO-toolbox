from datetime import datetime
from osgeo import gdal
import rioxarray as rio
from pathlib import Path
import os
import logging
from typing import List
import openeo

from eomaji.utils.process_era5 import process_single_date
from eomaji.utils.general_utils import load_lut

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

lut = load_lut()


def get_meteo_data(
    connection: openeo.rest.connection.Connection,
    date: str,
    bbox: List[float],
    data_dir: str = "./",
    cds_credentials_file=".cdsapirc",
    ads_credentials_file=".adsapirc",
):
    """
    Fetches meteorological and elevation data for a given date and bounding box

    Args:
        connection: OpenEO connection
        date (str): Date in 'YYYY-MM-DD' format.
        bbox (list): Bounding box as [west, south, east, north].
        out_dir (str): Output directory for saving the results. Default is './'.

    Returns:
        str: Path to the final output file, or None if an error occurred.
    """
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        logging.error("Invalid date format. Expected 'YYYY-MM-DD'.")
        return None

    if len(bbox) != 4:
        logging.error(
            "Bounding box must be a list of four coordinates: [west, south, east, north]."
        )
        return None

    aoi = dict(zip(["west", "south", "east", "north"], bbox))

    base_output_dir = os.path.join(data_dir, "meteo_data")
    date_dir = "/".join(date.split("-"))
    out_dir_meteo = os.path.join(base_output_dir, date_dir)
    os.makedirs(out_dir_meteo, exist_ok=True)
    # Paths for intermediate and final outputs
    dem_path = os.path.join(out_dir_meteo, "dem.tif")
    slope_path = os.path.join(out_dir_meteo, "slope.tif")
    aspect_path = os.path.join(out_dir_meteo, "aspect.tif")

    # Collection and band definitions
    s3_collection = "SENTINEL3_SLSTR_L2_LST"
    s3_band = ["LST"]
    s3_cube = connection.load_collection(
        s3_collection,
        spatial_extent=aoi,
        temporal_extent=date,
        bands=s3_band,
        properties={
            "timeliness": lambda x: x == "NT",
            "orbitDirection": lambda x: x == "DESCENDING",
        },
    )
    dem_cube = connection.load_collection("COPERNICUS_30", spatial_extent=aoi)
    resampled_dem_cube = dem_cube.resample_cube_spatial(s3_cube, method="bilinear")

    logging.info("Downloading and Resampling DEM for area")
    resampled_dem_cube.execute_batch(dem_path)

    logging.info("Processing slope and aspect from DEM.")
    gdal.DEMProcessing(slope_path, dem_path, "slope", computeEdges=True)
    gdal.DEMProcessing(
        aspect_path, dem_path, "aspect", computeEdges=True, zeroForFlat=True
    )

    # Extract date and acquisition time
    date_int = int(date.replace("-", ""))
    acq_time = float(
        s3_cube.metadata.temporal_dimension.extent[0].split("T")[1].split(":")[0]
    )

    # Call external processing function
    logging.info("Process era5 for single date'.")
    process_single_date(
        dem_path,
        slope_path,
        aspect_path,
        date_int,
        acq_time,
        out_dir_meteo,
        cds_credentials_file=cds_credentials_file,
        ads_credentials_file=ads_credentials_file,
    )

    # Rename output files for clarity
    logging.info("Renaming output files for clarity.")
    for f in os.listdir(out_dir_meteo):
        if "SW-IN-DD.tif" in f:
            os.rename(
                f"{out_dir_meteo}/{f}",
                f"{out_dir_meteo}/{f.replace('SW-IN-DD.tif', 'S_dn_24.tif')}",
            )
        if "TA.tif" in f:
            os.rename(
                f"{out_dir_meteo}/{f}",
                f"{out_dir_meteo}/{f.replace('TA.tif', 'T_A1.tif')}",
            )
        if "PA.tif" in f:
            os.rename(
                f"{out_dir_meteo}/{f}",
                f"{out_dir_meteo}/{f.replace('PA.tif', 'p.tif')}",
            )
        if "WS.tif" in f:
            os.rename(
                f"{out_dir_meteo}/{f}",
                f"{out_dir_meteo}/{f.replace('WS.tif', 'u.tif')}",
            )

    logging.info("Summing DI images.")
    di_files = list(Path(out_dir_meteo).glob("*-DI*"))
    if not di_files:
        logging.warning("No '-DI*' files found for summation.")
        return None

    sum_image = rio.open_rasterio(di_files[0])
    for file in di_files[1:]:
        image = rio.open_rasterio(file)
        sum_image += image

    # Save final output
    save_file = os.path.join(out_dir_meteo, file.stem.split("_")[0] + "_S_dn.tif")
    sum_image.rio.to_raster(save_file)
    logging.info(f"Final output saved to: {save_file}")

    return out_dir_meteo


def resample_meteo_to_s2(
    in_dir: str, out_dir: str, s2_path: str, nodata_value: int = -999
) -> None:
    """
    Resamples and reprojects meteorological data to match a Sentinel-2 image.

    Parameters:
    - in_dir (str): Directory containing input meteorological TIFF files.
    - out_dir (str): Directory to save the resampled TIFF files.
    - s2_path (str): Path to the Sentinel-2 reference image.
    - nodata_value (int, optional): Value to use for missing data. Default is -999.

    Returns:
    - None
    """
    logging.info("Starting resample_meteo_to_s2 processing...")

    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    product_list = ["EA.tif", "p.tif", "u.tif", "S_dn_24.tif", "S_dn.tif", "T_A1.tif"]
    tiff_files = [
        x
        for x in in_dir.glob("*.tif")
        if any(x.name.endswith(product) for product in product_list)
    ]

    if not tiff_files:
        logging.warning(f"No matching TIFF files found in {in_dir}. Exiting.")
        return

    try:
        high_res_ds = gdal.Open(s2_path)
        if high_res_ds is None:
            raise ValueError(f"Failed to open Sentinel-2 file: {s2_path}")

        high_res_proj = high_res_ds.GetProjection()
        high_res_geotransform = high_res_ds.GetGeoTransform()

        xmin = high_res_geotransform[0]
        xmax = xmin + high_res_geotransform[1] * high_res_ds.RasterXSize
        ymax = high_res_geotransform[3]
        ymin = ymax + high_res_geotransform[5] * high_res_ds.RasterYSize

        logging.info(f"Using Sentinel-2 extent: ({xmin}, {ymin}, {xmax}, {ymax})")

        for low_res in tiff_files:
            output_path = out_dir / low_res.name
            logging.info(f"Processing {low_res.name} -> {output_path}")

            gdal.Warp(
                str(output_path),
                str(low_res),
                format="GTiff",
                dstSRS=high_res_proj,
                xRes=high_res_geotransform[1],
                yRes=abs(high_res_geotransform[5]),
                resampleAlg="bilinear",
                srcNodata=None,
                dstNodata=nodata_value,
                outputBounds=(xmin, ymin, xmax, ymax),
            )

        logging.info("Processing completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        raise

from pathlib import Path
import os
import yaml
import numpy as np
import datetime as dt
from osgeo import gdal
from pyproj import Proj
import cdsapi
from meteo_utils import ecmwf_utils as eu
from meteo_utils import solar_irradiance as sun

METEO_DATA_FIELDS = ["TA", "EA", "WS", "PA", "AOT", "TCWV", "SW-IN", "LW-IN"]
CDS_VARIABLES = [
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_pressure",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "total_column_water_vapour",
    "geopotential",
]

ADS_VARIABLES = ["total_aerosol_optical_depth_550nm"]

DAILY_VARS = ["ETr", "SW-IN-DD"]


def process_single_date(
    elev_input_file,
    slope_input_file,
    aspect_input_file,
    date_int,
    acq_time,
    dst_folder=None,
    svf_input_file=None,
    blending_height=100,
    cds_credentials_file=os.path.join(os.path.expanduser("~"), ".cdsapirc"),
    ads_credentials_file=os.path.join(os.path.expanduser("~"), ".adsapirc"),
):
    """

    Parameters
    ----------
    elev_input_file : str
        Path to a GDAL compatible Digital Elevation Model
    slope_input_file : str
        Path to a GDAL compatible slope image (degrees)
    aspect_input_file : str
        Path to a GDAL compatible aspect image (0 for flat surfaces)
    date_int : int or str
        Acquisition date (YYYYMMDD)
    acq_time : float
        Acquistion time in decimal hour
    dst_folder : str, optional
        Path to the destination folder in which meteo products will be stored.
    svf_input_file : str, optional
        Path to a GDAL compatible Sky View Fraction image (0-1)
    blending_height : float, optional
        Elevation above ground level at which meteo products will be generated, default=100 magl

    Returns
    -------
    output : dict
        Dictionary of arrays with the output meteo products
    """
    with open(cds_credentials_file, "r") as f:
        credentials = yaml.safe_load(f)

    dst_folder = Path(dst_folder)
    print(f'Downloading "{", ".join(CDS_VARIABLES)}" from the Copernicus Climate Store')
    fid = gdal.Open(elev_input_file, gdal.GA_ReadOnly)
    gt = fid.GetGeoTransform()
    proj = fid.GetProjection()
    p = Proj(proj)
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * fid.RasterXSize
    miny = maxy + gt[5] * fid.RasterYSize
    del fid
    date_obj = dt.datetime.strptime(str(date_int), "%Y%m%d")
    date_ini = date_obj - dt.timedelta(1)
    date_end = date_obj + dt.timedelta(1)
    date_str = f"{date_ini.strftime('%Y-%m-%d')}/{date_end.strftime('%Y-%m-%d')}"
    # Area is North, West, South, East
    extent_geo = p(minx, maxy, inverse=True), p(maxx, miny, inverse=True)
    area = [
        extent_geo[0][1] + 1,
        extent_geo[0][0] - 1,
        extent_geo[1][1] - 1,
        extent_geo[1][0] + 1,
    ]
    print(
        f"Querying products for extent {area}\n"
        f"..and dates {date_obj - dt.timedelta(1)} to {date_obj + dt.timedelta(1)}"
    )

    # Connect to the server and download the data
    s = {
        "data_format": "grib",
        "variable": CDS_VARIABLES,
        "date": date_str,
        "area": area,
        "product_type": ["reanalysis"],
        "time": [str(t).zfill(2) + ":00" for t in range(0, 24, 1)],
    }

    cds_target = dst_folder / f"{date_int}_era5.grib"
    if not cds_target.exists():
        c = cdsapi.Client(
            url=credentials["url"], key=credentials["key"], quiet=True, progress=False
        )
        print(f"Saving into {cds_target}")
        c.retrieve("reanalysis-era5-single-levels", s, cds_target)
        print(f"Saved to file {cds_target}")

    # If we are working on the currrent year we need to query the forecast cams data, otherwise download reanalysis
    if date_obj.year == dt.date.today().year:
        dataset = "cams-global-atmospheric-composition-forecasts"
    else:
        dataset = "cams-global-reanalysis-eac4"

    ads_target = dst_folder / f"{date_int}_cams.grib"
    if not ads_target.exists():
        print(
            f'Downloading "{", ".join(ADS_VARIABLES)}" '
            f"from the Copernicus Atmospheric Store"
        )
        eu.download_ADS_data(
            dataset,
            date_obj - dt.timedelta(1),
            date_obj + dt.timedelta(1),
            ADS_VARIABLES,
            ads_target,
            overwrite=False,
            area=area,
            ads_credentials_file=ads_credentials_file,
        )
        print(f"Saved to file {ads_target}")

    date_obj = date_obj + dt.timedelta(hours=acq_time)
    time_zone = sun.angle_average(extent_geo[0][0], extent_geo[1][0]) / 15.0
    print(f"Processing ECMWF data for UTC time {date_obj}\nThis may take some time...")

    meteo_data_fields = METEO_DATA_FIELDS + DAILY_VARS
    output = eu.get_ECMWF_data(
        cds_target,
        date_obj,
        meteo_data_fields,
        elev_input_file,
        blending_height,
        aod550_data_file=ads_target,
        time_zone=time_zone,
        is_forecast=False,
        slope_file=slope_input_file,
        aspect_file=aspect_input_file,
        svf_file=svf_input_file,
    )

    out_dict = {}

    if dst_folder:
        for param, array in output.items():
            if param not in DAILY_VARS:
                hour = int(np.floor(acq_time))
                minute = int(60 * acq_time - hour)
                acq_time_str = f"{hour:02}{minute:02}"
                if param == "SW-IN":
                    for i, var1 in enumerate(["DIR", "DIF"]):
                        for j, var2 in enumerate(["PAR", "NIR"]):
                            param = f"{var2}-{var1}"
                            filename = f"{date_int}T{acq_time_str}_{param.upper()}.tif"
                            dst_file = str(dst_folder / filename)
                            print(f"Saving {param} to {dst_file}")
                            driver = gdal.GetDriverByName("MEM")
                            values = np.maximum(array[i][j], 0)
                            dims = values.shape
                            ds = driver.Create(
                                "MEM", dims[1], dims[0], 1, gdal.GDT_Float32
                            )
                            ds.SetProjection(proj)
                            ds.SetGeoTransform(gt)
                            ds.GetRasterBand(1).WriteArray(values)
                            driver_opt = [
                                "COMPRESS=DEFLATE",
                                "PREDICTOR=1",
                                "BIGTIFF=IF_SAFER",
                            ]
                            gdal.Translate(
                                dst_file,
                                ds,
                                format="GTiff",
                                creationOptions=driver_opt,
                                stats=True,
                            )
                else:
                    filename = f"{date_int}T{acq_time_str}_{param.upper()}.tif"
                    dst_file = str(dst_folder / filename)
                    print(f"Saving {param} to {dst_file}")
                    driver = gdal.GetDriverByName("MEM")
                    dims = array.shape
                    ds = driver.Create("MEM", dims[1], dims[0], 1, gdal.GDT_Float32)
                    ds.SetProjection(proj)
                    ds.SetGeoTransform(gt)
                    ds.GetRasterBand(1).WriteArray(array)
                    driver_opt = ["COMPRESS=DEFLATE", "PREDICTOR=1", "BIGTIFF=IF_SAFER"]
                    gdal.Translate(
                        dst_file,
                        ds,
                        format="GTiff",
                        creationOptions=driver_opt,
                        stats=True,
                    )
            else:
                filename = f"{date_int}_{param.upper()}.tif"
                dst_file = str(dst_folder / filename)
                print(f"Saving {param} to {dst_file}")
                driver = gdal.GetDriverByName("MEM")
                dims = array.shape
                ds = driver.Create("MEM", dims[1], dims[0], 1, gdal.GDT_Float32)
                ds.SetProjection(proj)
                ds.SetGeoTransform(gt)
                ds.GetRasterBand(1).WriteArray(array)
                driver_opt = ["COMPRESS=DEFLATE", "PREDICTOR=1", "BIGTIFF=IF_SAFER"]
                gdal.Translate(
                    dst_file, ds, format="GTiff", creationOptions=driver_opt, stats=True
                )

        del ds

    return output

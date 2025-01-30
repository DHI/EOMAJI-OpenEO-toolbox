import os
import logging
from shapely.geometry import box
from shapely import to_geojson
import rasterio
import numpy as np
from pathlib import Path
import xarray as xr

from eomaji.utils.general_utils import load_lut


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

lut = load_lut()


def get_biopar(connection, product, date, aoi):
    if isinstance(date, str):
        date = [date, date]
    biopar = connection.datacube_from_process(
        "BIOPAR",
        namespace="https://openeo.dataspace.copernicus.eu/openeo/1.1/processes/u:3e24e251-2e9a-438f-90a9-d4500e576574/BIOPAR",
        date=date,
        polygon=aoi,
        biopar_type=product,
    )
    return biopar


def calc_canopy(lai_path, worldcover_path, fg_path):
    with rasterio.open(lai_path) as lai_src:
        lai = lai_src.read(1).astype(np.float32)
        profile = lai_src.profile

    with rasterio.open(worldcover_path) as worldcover_src:
        landcover = worldcover_src.read(1).astype(np.int32)
        landcover = 10 * (landcover // 10)

    with rasterio.open(fg_path) as fg_src:
        fg = fg_src.read(1).astype(np.float32)

    param_value = np.ones(landcover.shape, np.float32) + np.nan

    for lc_class in np.unique(landcover[~np.isnan(landcover)]):
        lc_pixels = np.where(landcover == lc_class)
        lc_index = lut[lut["landcover_class"] == lc_class].index[0]
        param_value[lc_pixels] = lut["veg_height"][lc_index]
        if lut["is_herbaceous"][lc_index] == 1:
            pai = lai / fg
            pai = pai[lc_pixels]
            param_value[lc_pixels] = 0.1 * param_value[lc_pixels] + 0.9 * param_value[
                lc_pixels
            ] * np.minimum((pai / lut["veg_height"][lc_index]) ** 3.0, 1.0)

    output_path = lai_path.replace("LAI", "H_C")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(param_value, 1)
    print(f"Saved H_C to {output_path}")


def calc_fg(fapar_path, lai_path, sza_path):
    from pyTSEB import TSEB

    with rasterio.open(fapar_path) as fapar_src:
        fapar = fapar_src.read(1).astype(np.float32)
        profile = fapar_src.profile
    with rasterio.open(lai_path) as lai_src:
        lai = lai_src.read(1).astype(np.float32)

    with rasterio.open(sza_path) as sza_src:
        sza = sza_src.read(1).astype(np.float32)

    f_g = np.ones(lai.shape, np.float32)
    converged = np.zeros(lai.shape, dtype=bool)
    converged[np.logical_or(lai <= 0.2, fapar <= 0.1)] = True
    min_frac_green = 0.01

    for _ in range(50):
        f_g_old = f_g.copy()
        fipar = TSEB.calc_F_theta_campbell(
            sza[~converged], lai[~converged] / f_g[~converged], w_C=1, Omega0=1, x_LAD=1
        )
        f_g[~converged] = fapar[~converged] / fipar
        f_g = np.clip(f_g, min_frac_green, 1.0)
        converged = np.logical_or(np.isnan(f_g), np.abs(f_g - f_g_old) < 0.02)
        if np.all(converged):
            break

    profile.update(dtype=rasterio.float32, count=1)
    output_path = str(lai_path).replace("LAI", "F_G")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(f_g, 1)
    print(f"Saved frac_green to {output_path}")


def split_tifs(nc_file):
    out_dir = Path(nc_file).parent
    data = xr.open_dataset(nc_file)
    s2_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

    if Path(nc_file).stem == "s2_data":
        refl = data[s2_bands]
        refl = refl.rio.write_crs(data.crs.crs_wkt)
        date = str(data.t.values[0]).split("T")[0].replace("-", "")
        output_file = os.path.join(out_dir, f"{date}_REFL.tif")
        refl.sel(t=refl.t[0]).rio.to_raster(output_file)

    for var_name in data.data_vars:
        if var_name in s2_bands + ["crs"]:
            continue

        for t in data.t:
            band = data.sel(t=t)[var_name]
            if "grid_mapping" in band.attrs:
                del band.attrs["grid_mapping"]
            band = band.rio.write_crs(data.crs.crs_wkt)

            date = str(data.t.values[0]).replace("-", "").replace(":", "").split(".")[0]
            date = date.split("T")[0] if Path(nc_file).stem == "s2_data" else date

            if var_name == "viewZenithAngles":
                output_file = os.path.join(out_dir, f"{date}_VZA.tif")
            elif var_name == "sunZenithAngles":
                output_file = os.path.join(out_dir, f"{date}_SZA.tif")
            else:
                output_file = os.path.join(out_dir, f"{date}_{var_name}.tif")

            band.rio.to_raster(output_file)
        print(f"Saved {var_name} to {output_file}")


def _estimate_param_value(worldcover_path, lut, band, output_path):
    with rasterio.open(worldcover_path) as worldcover_src:
        landcover = worldcover_src.read(1).astype(np.int32)
        landcover = 10 * (landcover // 10)
        profile = worldcover_src.profile

    param_value = np.ones(landcover.shape) + np.nan

    for lc_class in np.unique(landcover[~np.isnan(landcover)]):
        lc_pixels = np.where(landcover == lc_class)
        lc_index = lut[lut["landcover_class"] == lc_class].index[0]
        param_value[lc_pixels] = lut[band][lc_index]

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(param_value, 1)

    print(f"Saved {band} to {output_path}")
    return param_value


def get_s2_data(connection, bbox, date, data_dir="./"):
    if isinstance(date, str):
        time_window = [date, date]
    if isinstance(date, list):
        time_window = date

    base_output_dir = os.path.join(data_dir, "s2_data")
    date_dir = "/".join(date.split("-"))
    out_dir = os.path.join(base_output_dir, date_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    bbox_polygon = eval(to_geojson(box(*bbox)))
    aoi = dict(zip(["west", "south", "east", "north"], bbox))

    fapar = get_biopar(connection, "FAPAR", time_window, bbox_polygon)
    lai = get_biopar(connection, "LAI", time_window, bbox_polygon)
    fcover = get_biopar(connection, "FCOVER", time_window, bbox_polygon)
    ccc = get_biopar(connection, "CCC", time_window, bbox_polygon)
    cwc = get_biopar(connection, "CWC", time_window, bbox_polygon)

    s2_collection = "SENTINEL2_L2A"
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

    s2_cube = connection.load_collection(
        s2_collection, spatial_extent=aoi, temporal_extent=time_window, bands=s2_bands
    )

    lst_cube = connection.load_collection(
        "SENTINEL3_SLSTR_L2_LST",
        spatial_extent=aoi,
        temporal_extent=time_window,
        bands=["LST"],
        properties={
            "timeliness": lambda x: x == "NT",
            "orbitDirection": lambda x: x == "DESCENDING",
        },
    )

    vza_cube = connection.load_collection(
        "SENTINEL3_SLSTR_L2_LST",
        spatial_extent=aoi,
        temporal_extent=time_window,
        bands=["viewZenithAngles"],
        properties={
            "timeliness": lambda x: x == "NT",
            "orbitDirection": lambda x: x == "DESCENDING",
        },
    )

    dem_cube = connection.load_collection("COPERNICUS_30", spatial_extent=aoi)

    worldcover = connection.load_collection(
        "ESA_WORLDCOVER_10M_2021_V2", temporal_extent=["2021-01-01", "2021-12-31"]
    ).filter_bbox(bbox)

    biopar_cube = fapar.merge_cubes(lai)
    biopar_cube = biopar_cube.merge_cubes(fcover)
    biopar_cube = biopar_cube.merge_cubes(ccc)
    biopar_cube = biopar_cube.merge_cubes(cwc)
    s2_full_cube = biopar_cube.merge_cubes(s2_cube)
    dem_resampled_s2_cube = dem_cube.resample_cube_spatial(
        s2_full_cube, method="bilinear"
    )
    wc_resampled_s2_cube = worldcover.resample_cube_spatial(
        s2_full_cube, method="bilinear"
    )
    lst_resampled_cube = lst_cube.resample_cube_spatial(s2_full_cube, method="bilinear")
    vza_resampled_cube = vza_cube.resample_cube_spatial(s2_full_cube, method="bilinear")

    s2_path = os.path.join(out_dir, "s2_data.nc")
    lst_path = os.path.join(out_dir, "lst_data.nc")
    vza_path = os.path.join(out_dir, "vza_data.nc")

    if not os.path.exists(s2_path):
        s2_full_cube.execute_batch(s2_path)
    split_tifs(s2_path)

    datestr = date.replace("-", "")
    dem_path = os.path.join(out_dir, f"{datestr}_ELEV.tif")
    if not os.path.exists(dem_path):
        dem_resampled_s2_cube.execute_batch(dem_path)

    worldcover_path = os.path.join(out_dir, f"WordlCover2021.tif")
    if not os.path.exists(worldcover_path):
        wc_resampled_s2_cube.execute_batch(worldcover_path)

    if not os.path.exists(lst_path):
        lst_resampled_cube.execute_batch(lst_path)

    if not os.path.exists(vza_path):
        vza_resampled_cube.execute_batch(vza_path)

    split_tifs(lst_path)
    split_tifs(vza_path)

    lai_path = os.path.join(out_dir, f"{datestr}_LAI.tif")
    fapar_path = os.path.join(out_dir, f"{datestr}_FAPAR.tif")
    sza_path = os.path.join(out_dir, f"{datestr}_SZA.tif")
    fg_path = os.path.join(out_dir, f"{datestr}_F_G.tif")

    calc_fg(fapar_path, lai_path, sza_path)
    calc_canopy(lai_path, worldcover_path, fg_path)

    out_path = os.path.join(out_dir, f"{datestr}_W_C.tif")
    _ = _estimate_param_value(worldcover_path, lut, "veg_height_width_ratio", out_path)
    out_path = os.path.join(out_dir, f"{datestr}_LEAF_WIDTH.tif")
    _ = _estimate_param_value(worldcover_path, lut, "veg_leaf_width", out_path)

    return (
        (
            s2_full_cube,
            dem_resampled_s2_cube,
            wc_resampled_s2_cube,
            lst_resampled_cube,
            vza_resampled_cube,
        ),
        out_dir,
    )


def watercloud_model(param, a, b, c):
    result = a + b * (1.0 - np.exp(c * param))

    return result


def cab_to_vis_spectrum(
    cab,
    coeffs_wc_rho_vis=[0.14096573, -0.09648072, -0.06328343],
    coeffs_wc_tau_vis=[0.08543707, -0.08072709, -0.06562554],
):
    rho_leaf_vis = watercloud_model(cab, *coeffs_wc_rho_vis)
    tau_leaf_vis = watercloud_model(cab, *coeffs_wc_tau_vis)

    rho_leaf_vis = np.clip(rho_leaf_vis, 0, 1)
    tau_leaf_vis = np.clip(tau_leaf_vis, 0, 1)

    return rho_leaf_vis, tau_leaf_vis


def cw_to_nir_spectrum(
    cw,
    coeffs_wc_rho_nir=[0.38976106, -0.17260689, -65.7445699],
    coeffs_wc_tau_nir=[0.36187620, -0.18374560, -65.3125878],
):
    rho_leaf_nir = watercloud_model(cw, *coeffs_wc_rho_nir)
    tau_leaf_nir = watercloud_model(cw, *coeffs_wc_tau_nir)

    rho_leaf_nir = np.clip(rho_leaf_nir, 0, 1)
    tau_leaf_nir = np.clip(rho_leaf_nir, 0, 1)

    return rho_leaf_nir, tau_leaf_nir


def process_lai_to_vis(lai_path):
    """Processes a LAI raster file to generate visible spectrum reflectance and transmittance TIFFs."""
    try:
        if not Path(lai_path).exists():
            raise FileNotFoundError(f"LAI file not found: {lai_path}")

        with rasterio.open(lai_path) as src:
            meta = src.meta.copy()
            meta.update(dtype="float32")

            lai = src.read(1)
            cab = np.clip(np.array(lai), 0.0, 140.0)
            refl_vis, trans_vis = cab_to_vis_spectrum(cab)  # Function assumed to exist

            rho_vis_path = lai_path.replace("LAI", "RHO_VIS_C")
            tau_vis_path = lai_path.replace("LAI", "TAU_VIS_C")

            save_raster(rho_vis_path, refl_vis, meta)
            save_raster(tau_vis_path, trans_vis, meta)

            logging.info(f"Processed LAI to VIS: {rho_vis_path}, {tau_vis_path}")

    except Exception as e:
        logging.error(f"Error processing LAI: {e}")
        raise  # Ensure function fails on error


def process_cwc_to_nir(cw_path):
    """Processes a CWC raster file to generate NIR reflectance and transmittance TIFFs."""
    try:
        if not Path(cw_path).exists():
            raise FileNotFoundError(f"CWC file not found: {cw_path}")

        with rasterio.open(cw_path) as src:
            meta = src.meta.copy()
            meta.update(dtype="float32")

            cw = src.read(1)
            cw = np.clip(np.array(cw), 0.0, 0.1)
            refl_nir, trans_nir = cw_to_nir_spectrum(cw)  # Function assumed to exist

            rho_nir_path = cw_path.replace("CWC", "RHO_NIR_C")
            tau_nir_path = cw_path.replace("CWC", "TAU_NIR_C")

            save_raster(rho_nir_path, refl_nir, meta)
            save_raster(tau_nir_path, trans_nir, meta)

            logging.info(f"Processed CWC to NIR: {rho_nir_path}, {tau_nir_path}")

    except Exception as e:
        logging.error(f"Error processing CWC: {e}")
        raise  # Ensure function fails on error


def save_raster(output_path, data, meta):
    """Saves an array as a GeoTIFF using Rasterio."""
    try:
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(data.astype("float32"), 1)
        logging.info(f"Saved raster: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save raster {output_path}: {e}")
        raise  # Ensure function fails on error


def process_lai_and_cwc(lai_path, cw_path):
    process_lai_to_vis(lai_path)
    process_cwc_to_nir(cw_path)


def save_lat_lon_as_tifs(nc_file, out_dir, date):
    data = xr.open_dataset(nc_file)

    lat, lon = xr.broadcast(data["y"], data["x"])

    lat = lat.rio.write_crs(data.crs.crs_wkt)
    lat.rio.to_raster(f"{out_dir}/{date}_LAT.tif")

    lon = lon.rio.write_crs(data.crs.crs_wkt)
    lon.rio.to_raster(f"{out_dir}/{date}_LON.tif")

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:25:09 2018

@author: rmgu
"""

import math
import numpy as np
import numpy.ma as ma
import os
from pathlib import Path
import shutil
import tempfile

from osgeo import ogr, osr, gdal
import xarray

from pyDMS.pyDMSUtils import getRasterInfo, openRaster, resampleWithGdalWarp


# Replacement strings for VRT template used to wrap Sentinel-3 data
RASTER_X_SIZE = "<RASTER_X_SIZE>"
RASTER_Y_SIZE = "<RASTER_Y_SIZE>"
X_DATASET = "<X_DATASET>"
Y_DATASET = "<Y_DATASET>"
SOURCE_FILENAME = "<SOURCE_FILENAME>"


def raster_data(raster, bands=1, mask_nodata=True, subset_pix=None):
    fid, closeOnExit = openRaster(raster)
    if type(bands) == int:
        bands = [bands]

    data = None
    for band in bands:
        scale = fid.GetRasterBand(band).GetScale() or 1
        offset = fid.GetRasterBand(band).GetOffset() or 0
        if subset_pix is None:
            band_data = fid.GetRasterBand(band).ReadAsArray().astype(np.float32)
        else:
            band_data = (
                fid.GetRasterBand(band)
                .ReadAsArray(
                    subset_pix[0][1],
                    subset_pix[0][0],
                    subset_pix[1][1] - subset_pix[0][1],
                    subset_pix[1][0] - subset_pix[0][0],
                )
                .astype(np.float32)
            )
        band_data = band_data * scale + offset
        no_data_value = raster_nodata_value(fid)
        if no_data_value is not None:
            band_data[band_data == no_data_value * scale + offset] = no_data_value
        if data is None:
            data = band_data
        else:
            data = np.dstack((data, band_data))

    if mask_nodata:
        nodata_value = raster_nodata_value(fid)
        if nodata_value is None and np.any(np.isnan(data)):
            nodata_value = np.nan
        if nodata_value is None:
            data = ma.asarray(data)
        elif np.isnan(nodata_value):
            data = ma.masked_invalid(data)
        else:
            data = ma.masked_equal(data, nodata_value)

    if closeOnExit:
        fid = None

    return data


def raster_info(raster):
    return getRasterInfo(raster)


def raster_nodata_value(raster):
    fid, closeOnExit = openRaster(raster)
    nodata_value = fid.GetRasterBand(1).GetNoDataValue()
    if closeOnExit:
        fid = None
    return nodata_value


def save_image(
    data,
    geotransform,
    projection,
    filename,
    should_scale=True,
    nodata_value=None,
    dst_no_data_non_mem=-9999,
    fieldNames=[],
):
    """
    Saves raster data to a file or in memory with specified geospatial metadata.
    If the file is saved to disk the file is a COG and its data are scaled and converted to
    integers

    Parameters:
    data (numpy.ndarray): A multidimensional numpy array containing the raster data to be saved.
    geotransform (tuple): A tuple in GDAL geotransform format of six values representing the affine
        transformation parameters.
        Format: (top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel
                 resolution)
    projection (str): A string representation of the spatial reference (WKT or EPSG code).
    filename (str): The file path where the raster will be saved. If MEM then the file is kept in
        memory only.
    should_scale(bool, optional): True is values should be scaled by 10000 and result saved as
        int32, False otherwise. Defaults to True.
    nodata_value (int or float, optional): The value that represents no data in the raster. Default
        is None.
    dst_no_data_non_mem (int, optional): The value to be written in the saved file as nodata. It is
        only used when the file is saved in disk Default is -9999.
    fieldNames (list, optional): A list of field names to be included in the metadata in case the
        resulting file is a NetCDF. Default is an empty list.

    Returns:
    gdal.Dataset: A GDAL dataset object representing the saved raster.
    """

    if ma.isMaskedArray(data) and nodata_value is not None:
        data = data.filled(nodata_value)
    data = data.astype(np.float32)
    """
    # GDAL has problem saving COG overviews when nodata is NaN, so change it if needed
    if ((nodata_value is None) or np.isnan(nodata_value)) and filename.endswith(".tif"):
        if data.dtype == np.bool:
            data = data.astype(np.byte)
        try:
            new_nodata_value = np.finfo(data.dtype).max
        except ValueError:
            new_nodata_value = np.iinfo(data.dtype).max
        data[np.isnan(data)] = new_nodata_value
        nodata_value = new_nodata_value
    """

    filename = str(filename)

    if filename == "MEM" or should_scale == False:
        dtype = gdal.GDT_Float32
        scale = 1
        no_data_value = nodata_value
    else:
        dtype = gdal.GDT_Int32

        # Scale
        data[data == nodata_value] = np.nan
        if np.nanmax(data) * 10_000 <= 2147483647:
            reverse_scale = 10_000
            scale = 1 / reverse_scale
            data = data * reverse_scale
            print(f"Scaling by {reverse_scale}")
        elif np.nanmax(data) * 1_000 <= 2147483647:
            reverse_scale = 1_000
            scale = 1 / reverse_scale
            data = data * reverse_scale
            print(f"Scaling by {reverse_scale}")
        else:
            scale = 1

        # Assign nodata value
        data[np.isnan(data)] = dst_no_data_non_mem
        no_data_value = dst_no_data_non_mem
        data = data.astype(np.int32)

    # Save to memory first
    is_netCDF = False
    memDriver = gdal.GetDriverByName("MEM")
    shape = data.shape
    if len(shape) > 2:
        ds = memDriver.Create("MEM", shape[1], shape[0], shape[2], dtype)
        ds.SetProjection(projection)
        ds.SetGeoTransform(geotransform)
        for i in range(shape[2]):
            ds.GetRasterBand(i + 1).WriteArray(data[:, :, i])
            ds.GetRasterBand(i + 1).SetScale(scale)
            if filename == "MEM":
                if nodata_value is None:
                    nodata_value = np.nan
            if nodata_value is not None:
                ds.GetRasterBand(i).SetNoDataValue(nodata_value)
    else:
        ds = memDriver.Create("MEM", shape[1], shape[0], 1, dtype)
        ds.SetProjection(projection)
        ds.SetGeoTransform(geotransform)
        ds.GetRasterBand(1).WriteArray(data)
        ds.GetRasterBand(1).SetScale(scale)
        if filename == "MEM":
            if nodata_value is None:
                nodata_value = np.nan
        if nodata_value is not None:
            ds.GetRasterBand(1).SetNoDataValue(nodata_value)

    # Save to file if required
    if filename != "MEM":
        # If the output file has .nc extension then save it as netCDF,
        # otherwise assume that the output should be a GeoTIFF (COG)
        ext = os.path.splitext(filename)[1]
        if ext.lower() == ".nc":
            fileFormat = "netCDF"
            driverOpt = ["FORMAT=NC2"]
            is_netCDF = True
        else:
            fileFormat = "COG"
            driverOpt = [
                "COMPRESS=ZSTD",
                "PREDICTOR=2",
                "BIGTIFF=IF_SAFER",
            ]

        print(f"Attempting to save {filename} as {fileFormat}")
        ds = gdal.Translate(
            filename,
            ds,
            format=fileFormat,
            creationOptions=driverOpt,
            noData=no_data_value,
            stats=True,
        )
        if ds is None:
            raise Exception("Failed to save {filename} with {fileFormat} driver")

        # In case of netCDF format use netCDF4 module to assign proper names
        # to variables (GDAL can't do this). Also it seems that GDAL has
        # problems assigning projection to all the bands so fix that.
        if is_netCDF and fieldNames:
            from netCDF4 import Dataset

            ds = None
            ds = Dataset(filename, "a")
            grid_mapping = ds["Band1"].grid_mapping
            for i, field in enumerate(fieldNames):
                ds.renameVariable("Band" + str(i + 1), field)
                ds[field].grid_mapping = grid_mapping
            ds.close()
            ds = gdal.Open('NETCDF:"' + filename + '":' + fieldNames[0])

    print("Saved " + filename)

    return ds


def resample_with_gdalwarp(src, template, resample_alg):
    return resampleWithGdalWarp(src, template, resampleAlg=resample_alg)


def merge_raster_layers(input_list, output_filename, separate=False, geotiff=False):
    merge_list = []
    nodata_str = ""
    for input_file in input_list:
        nodata_value = raster_nodata_value(input_file)
        if nodata_value is None:
            nodata_value = np.nan
        bands = raster_info(input_file)[5]
        # GDAL Build VRT cannot stack multiple multi-band images, so they have to be split into
        # multiple singe-band images first.
        if bands > 1:
            for band in range(1, bands + 1):
                temp_filename = tempfile.mkstemp(suffix="_" + str(band) + ".vrt")[1]
                gdal.BuildVRT(temp_filename, [input_file], bandList=[band])
                merge_list.append(temp_filename)
                nodata_str = nodata_str + f" {nodata_value}"
        else:
            merge_list.append(input_file)
            nodata_str = nodata_str + f" {nodata_value}"

    if geotiff:
        temp_filename = tempfile.mkstemp(suffix="_temp.vrt")[1]
        gdal.BuildVRT(
            temp_filename, merge_list, separate=separate, VRTNodata=nodata_str
        )
        fp = gdal.Translate(
            output_filename,
            temp_filename,
            format="COG",
            creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=2"],
        )
    else:
        fp = gdal.BuildVRT(
            output_filename, merge_list, separate=separate, VRTNodata=nodata_str
        )
    return fp


def get_subset(roi_shape, raster_proj_wkt, raster_geo_transform):
    # Find extent of ROI in roiShape projection
    roi = ogr.Open(roi_shape)
    roi_layer = roi.GetLayer()
    roi_extent = roi_layer.GetExtent()

    # Convert the extent to raster projection
    roi_proj = roi_layer.GetSpatialRef()
    raster_proj = osr.SpatialReference()
    raster_proj.ImportFromWkt(raster_proj_wkt)
    point_UL = convert_coordinate(
        (min(roi_extent[0], roi_extent[1]), max(roi_extent[2], roi_extent[3])),
        roi_proj,
        raster_proj,
    )
    point_LR = convert_coordinate(
        (max(roi_extent[0], roi_extent[1]), min(roi_extent[2], roi_extent[3])),
        roi_proj,
        raster_proj,
    )

    # Get pixel location of this extent
    ulX = raster_geo_transform[0]
    ulY = raster_geo_transform[3]
    pixel_size = raster_geo_transform[1]
    pixel_UL = [
        max(int(math.floor((ulY - point_UL[1]) / pixel_size)), 0),
        max(int(math.floor((point_UL[0] - ulX) / pixel_size)), 0),
    ]
    pixel_LR = [
        int(round((ulY - point_LR[1]) / pixel_size)),
        int(round((point_LR[0] - ulX) / pixel_size)),
    ]

    # Get projected extent
    point_proj_UL = (ulX + pixel_UL[1] * pixel_size, ulY - pixel_UL[0] * pixel_size)
    point_proj_LR = (ulX + pixel_LR[1] * pixel_size, ulY - pixel_LR[0] * pixel_size)

    # Create a subset from the extent
    subset_proj = [point_proj_UL, point_proj_LR]
    subset_pix = [pixel_UL, pixel_LR]

    return subset_pix, subset_proj


def read_subset(source, subset_pix):
    if type(source) is np.ndarray or type(source) is ma.masked_array:
        data = source[
            subset_pix[0][0] : subset_pix[1][0], subset_pix[0][1] : subset_pix[1][1]
        ]
    elif type(source) == int or type(source) == float:
        data = (
            np.zeros(
                (
                    subset_pix[1][0] - subset_pix[0][0],
                    subset_pix[1][1] - subset_pix[0][1],
                )
            )
            + source
        )
    # Otherwise it should be a file path
    else:
        data = raster_data(source, bands=1, mask_nodata=False, subset_pix=subset_pix)
    return data


# Save pyTSEB input dataset to an NetCDF file
def save_dataset(
    dataset,
    gt,
    proj,
    output_filename,
    roi_vector=None,
    attrs={},
    compression={"zlib": True, "complevel": 6},
):
    # Get the raster subset extent
    if roi_vector is not None:
        subset_pix, subset_proj = get_subset(roi_vector, proj, gt)
        # NetCDF and GDAL geocoding are off by half a pixel so need to take this into account.
        pixel_size = gt[1]
        subset_proj = [
            [
                subset_proj[0][0] + 0.5 * pixel_size,
                subset_proj[0][1] - 0.5 * pixel_size,
            ],
            [
                subset_proj[1][0] + 0.5 * pixel_size,
                subset_proj[1][1] - 0.5 * pixel_size,
            ],
        ]
    else:
        shape = dataset[list(dataset)[0]].shape
        subset_pix = [[0, 0], [shape[0], shape[1]]]
        subset_proj = [
            [gt[0] + gt[1] * 0.5, gt[3] + gt[5] * 0.5],
            [gt[0] + gt[1] * (shape[1] + 0.5), gt[3] + gt[5] * (shape[0] + 0.5)],
        ]

    # Create xarray DataSet
    x = np.linspace(
        subset_proj[0][1],
        subset_proj[1][1],
        subset_pix[1][0] - subset_pix[0][0],
        endpoint=False,
    )
    y = np.linspace(
        subset_proj[0][0],
        subset_proj[1][0],
        subset_pix[1][1] - subset_pix[0][1],
        endpoint=False,
    )
    ds = xarray.Dataset(
        {}, coords={"x": (["x"], x), "y": (["y"], y), "crs": (["crs"], [])}
    )
    ds.crs.attrs["spatial_ref"] = proj

    # Save the data in the DataSet
    encoding = {}
    for name in dataset:
        data = read_subset(dataset[name], subset_pix)
        ds = ds.assign(
            temporary=xarray.DataArray(
                data, coords=[ds.coords["x"], ds.coords["y"]], dims=("x", "y")
            )
        )
        ds["temporary"].attrs["grid_mapping"] = "crs"
        ds = ds.rename({"temporary": name})
        encoding[name] = compression

    ds.attrs = attrs

    # Save dataset to file
    ds.to_netcdf(output_filename, encoding=encoding)


def prj_to_src(prj):
    src = osr.SpatialReference()
    src.ImportFromWkt(prj)
    return src


def prj_to_epsg(prj):
    src = osr.SpatialReference()
    src.ImportFromWkt(prj)
    epsg = int(src.GetAttrValue("AUTHORITY", 1))
    return epsg


def get_map_coordinates(row, col, geoTransform):
    X = geoTransform[0] + geoTransform[1] * col + geoTransform[2] * row
    Y = geoTransform[3] + geoTransform[4] * col + geoTransform[5] * row
    return X, Y


def get_pixel_coordinates(X, Y, geoTransform):
    row = (Y - geoTransform[3]) / geoTransform[5]
    col = (X - geoTransform[0]) / geoTransform[1]
    return int(row), int(col)


def convert_coordinate(input_coordinate, input_src, output_src=None, Z_in=0):
    """Coordinate conversion between two coordinate systems

    Parameters
    ----------
    input_coordinate : tuple
        input coordinate (x,y)
    inputEPSG : int
        EPSG coordinate code of input coordinates
    outputEPSG : int
       EPSG coordinate code of output coordinates
    Z_in : float
        input altitude, default=0

    Returns
    -------
    X_out : float
        output X coordinate
    Y_out : float
        output X coordinate
    Z_out : float
        output X coordinate
    """

    if not output_src:
        output_src = osr.SpatialReference()
        output_src.ImportFromEPSG(4326)

    try:
        # For GDAL 3 indicate the "legacy" axis order
        # https://gdal.org/tutorials/osr_api_tut.html#crs-and-axis-order
        # https://github.com/OSGeo/gdal/issues/1546
        input_src = input_src.Clone()
        input_src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        output_src = output_src.Clone()
        output_src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except AttributeError:
        # For GDAL 2 do nothing
        pass

    # create coordinate transformation
    coordTransform = osr.CoordinateTransformation(input_src, output_src)

    # transform point
    X_out, Y_out, Z_out = coordTransform.TransformPoint(
        input_coordinate[0], input_coordinate[1], Z_in
    )

    # print point in EPSG 4326
    return X_out, Y_out, Z_out


# Use geolocation arrays to warp and subset Sentinel-3 data.
def warp_sentinel3(
    data_src,
    latitude_src,
    longitude_src,
    output_filename,
    extent=None,
    template_file=None,
    resample_alg="Near",
):
    def _covert_to_tif(out_file, in_src):
        gdal.Translate(
            out_file,
            f'NETCDF:"{in_src["file"]}":{in_src["layer"]}',
            unscale=True,
            outputType=gdal.GDT_Float32,
        )

    def _prepare_vrt_file(vrt_file, replacement_strings, replacement_values):
        vrt_file_template = Path(__file__).parent / "ancillary" / "S3_template.vrt"
        with open(vrt_file_template, "r") as template, open(vrt_file, "w") as config:
            contents = template.read()
            for i, replacementString in enumerate(replacement_strings):
                contents = contents.replace(replacementString, replacement_values[i])
            config.write(contents)

    # Create temporary geotiffs of latitude, longitude and data. Otherwise scale_factor and
    # add_offset are ignored
    temp_dir = Path(output_filename).parent / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    longitude_file = str(temp_dir / "longitude.tif")
    _covert_to_tif(longitude_file, longitude_src)
    latitude_file = str(temp_dir / "latitude.tif")
    _covert_to_tif(latitude_file, latitude_src)
    data_file = str(temp_dir / "data.tif")
    _covert_to_tif(data_file, data_src)

    # Fill in the VRT template
    x_size, y_size = raster_info(data_file)[2:4]
    vrt_file = str(temp_dir / "temp.vrt")
    _prepare_vrt_file(
        vrt_file,
        [RASTER_X_SIZE, RASTER_Y_SIZE, X_DATASET, Y_DATASET, SOURCE_FILENAME],
        [str(x_size), str(y_size), longitude_file, latitude_file, data_file],
    )

    # Warp the VRT either to projection, extent and resolution of template file or to geographic
    # corrdinates
    if template_file:
        proj, gt, _, _, extent, _ = raster_info(template_file)
        gdal.Warp(
            output_filename,
            vrt_file,
            resampleAlg=resample_alg,
            geoloc=True,
            dstSRS=proj,
            xRes=gt[1],
            yRes=gt[5],
            outputBounds=extent,
            multithread=True,
            format="COG",
            creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=2"],
        )
    else:
        # First warp to geographic projection, then subset. Otherwise the resolution changes
        # depending on the subset.
        temp = gdal.Warp(
            "",
            vrt_file,
            dstSRS="EPSG:4326",
            resampleAlg=resample_alg,
            geoloc=True,
            multithread=True,
            format="MEM",
        )
        if extent is None:
            gdal.Translate(
                output_filename,
                temp,
                format="COG",
                creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=2"],
            )
        else:
            gdal.Translate(
                output_filename,
                temp,
                projWin=[extent[0], extent[3], extent[2], extent[1]],
                format="COG",
                creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=2"],
            )

    # Clean up
    shutil.rmtree(temp_dir)

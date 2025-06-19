![Project Status](https://img.shields.io/badge/Status-Development-yellow)
[![GitHub last commit](https://img.shields.io/github/last-commit/DHI/EOMAJI-OpenEO-toolbox)](#)

# EOMAJI OpenEO Toolbox
> ⚠️ **Note:** This project is under active development. Features may change and bugs may exist.

This repository contains openEO workflows for various python modules used in Evapotranspiration (ET) modeling and Irrigation mapping. 
Following modules are currently included:
* [Data Mining Sharpener (pyDMS)](https://github.com/radosuav/pyDMS)
* [Two Source Energy Balance (TSEB)](https://github.com/hectornieto/pyTSEB)


## Installation
To install the EOMAJI OpenEO Toolbox locally, follow these steps:

1. Install GDAL
Make sure GDAL is installed on your system. This is a required dependency for geospatial data processing.

2. Install the Toolbox from GitHub
Once GDAL is installed, you can install the toolbox directly using pip:
    ```
    pip install eomaji@git+https://github.com/DHI/EOMAJI-OpenEO-toolbox.git
    ```
>💡**Note**: If you're using the notebooks on Copernicus Data Space (CDSE) JupyterHub, the package is installed in the first cell. You *should* be able run the notebooks out of the box without any additional setup if you choose the "Geo Science Kernel"

## Running the Evapotranspiration Workflow  
The **notebooks** provided in the [`notebooks/`](./notebooks) folder demonstrate how to use the **EOMAJI OpenEO toolbox** for evapotranspiration modeling.  

These notebooks can be run directly on **Copernicus Data Space (CDSE) JupyterHub** for efficient processing and scalability.  

### Available Notebooks  
- **[`notebooks/step1_prepare_data.ipynb`](./notebooks/step1_prepare_data.ipynb)** – This notebook prepares Sentinel-2 and Sentinel-3 data for PyDMS and ET Flows by allowing users to define an area of interest, select suitable acquisition dates, and download the relevant datasets using OpenEO. It extracts vegetation indices and land cover parameters from **Sentinel-2** and **ESA WorldCover** datasets. 

- **[`notebooks/step2_pydms.ipynb`](./notebooks/step2_pydms.ipynb)** – Demonstrates how to use the **Data Mining Sharpener (pyDMS)** to refine Sentinel-3 Land Surface Temperature (LST) using Sentinel-2 reflectance data.  

- **[`notebooks/step3_et_input_parameters.ipynb`](./notebooks/step3_et_input_parameters.ipynb)** – Focuses on preprocessing meteorological and biophysical input data. This includes:  
  - Retrieving meteorological parameters from the **Copernicus Climate Data Store (CDS)**.  
  - Resampling the meteorological parameters to the Sentinel 2 resolution

- **[`notebooks/step4_et_tseb.ipynb`](./notebooks/step4_et_tseb.ipynb)** – Runs the **Two Source Energy Balance (TSEB)** model to estimate evapotranspiration. It takes as input:  
  - Sharpened LST from pyDMS.  
  - Preprocessed meteorological and vegetation parameters.  

These notebooks form a complete workflow, from data retrieval and preprocessing to sharpening LST and running the ET model.

## Development
You are welcome to contribute to the project my making either a Pull Request or a Ticket.

For setting up a development environment, you have two options:
1. **Using a Dev Container**
    This repository includes a devcontainer setup, which provides a pre-configured environment for development.

2. **Manual Setup** If you prefer a local setup
    * Make sure GDAL is installed on your system.
    * Create a virtual environment and install the package with either pip or UV:
    ```sh
        python -m venv eomaji-env
        source eomaji-env/bin/activate # On Windows, use `eomaji-env\Scripts\activate`
        pip install .
    ```
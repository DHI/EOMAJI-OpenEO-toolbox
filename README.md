# EOMAJI OpenEO Toolbox
This repository contains openEO workflows for various python modules used in Evapotranspiration (ET) modeling and Irrigation mapping. 

Following modules are currently included:
* [Data Mining Sharpener (pyDMS)](https://github.com/radosuav/pyDMS)
* [Two Source Energy Balance (TSEB)](https://github.com/hectornieto/pyTSEB)

## Installation
* first install GDAL on your machine
* Then install the package from Github
    ```
    pip install eomaji@git+https://github.com/DHI/EOMAJI-OpenEO-toolbox.git
    ```

## Running the Evapotranspiration Workflow  

The **notebooks** provided in the [`notebooks/`](./notebooks) folder demonstrate how to use the **EOMAJI OpenEO toolbox** for evapotranspiration modeling.  

These notebooks can be run directly on **Copernicus Data Space (CDSE) JupyterHub** for efficient processing and scalability.  

### Available Notebooks  

- **[`notebooks/pydms_example.ipynb`](./notebooks/pydms_example.ipynb)** – Demonstrates how to use the **Data Mining Sharpener (pyDMS)** to refine Sentinel-3 Land Surface Temperature (LST) using Sentinel-2 reflectance data.  

- **[`notebooks/et_input_parameters.ipynb`](./notebooks/et_input_parameters.ipynb)** – Focuses on downloading and preprocessing meteorological and biophysical input data. This includes:  
  - Retrieving meteorological parameters from the **Copernicus Climate Data Store (CDS)**.  
  - Extracting vegetation indices and land cover parameters from **Sentinel-2** and **ESA WorldCover** datasets.  

- **[`notebooks/et_tseb.ipynb`](./notebooks/et_tseb.ipynb)** – Runs the **Two Source Energy Balance (TSEB)** model to estimate evapotranspiration. It takes as input:  
  - Sharpened LST from pyDMS.  
  - Preprocessed meteorological and vegetation parameters.  

These notebooks form a complete workflow, from data retrieval and preprocessing to sharpening LST and running the ET model.



## Development
For development, you can set up the environment in one of two ways:

1. **Using a Dev Container**
    This repository includes a devcontainer setup, which provides a pre-configured environment for development.

2. **Manual Setup** If you prefer a local setup
    * Install GDAL
    * Create a virtual environment and install the package:
    ```sh
        python -m venv eomaji-env
        source eomaji-env/bin/activate # On Windows, use `eomaji-env\Scripts\activate`
        pip install .
    ```
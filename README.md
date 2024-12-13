# EOMAJI OpenEO Toolbox
This repository contains openEO workflows for various python modules used in Evapotranspiration (ET) modeling and Irrigation mapping. 

Following modules are currently included:
* [Data Mining Sharpener (pyDMS)](https://github.com/radosuav/pyDMS)
* .. more to come

## Installation
* first install GDAL on your machine
* Then install the package from Github
    ```
    pip install eomaji@git+https://github.com/DHI/EOMAJI-OpenEO-toolbox.git
    ```

## Workflow Examples
In the [notebooks](./notebooks) folder you can find examples of how to run the modules on OpenEO Datacubes. All these examples can be run on Copernicus Dataspace the jupyterhub
* In [notebooks/pydms_example.ipynb](./notebooks/pydms_example.ipynb) you find the example of running the Data Mining Sharpener on Sentinel3 LST using Sentinel2  


## Development
For development you can either
* Use the [devcontainer](./.devcontainer)  
* Install GDAL on you machine and then install the python package in a virtual environment

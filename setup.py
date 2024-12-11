import subprocess
import sys
from setuptools import setup


def get_gdal_version():
    """Fetch GDAL version using gdal-config."""
    try:
        version = (
            subprocess.check_output(["gdal-config", "--version"])
            .strip()
            .decode("utf-8")
        )
        return version
    except subprocess.CalledProcessError:
        print("Error: gdal-config not found or not working properly.")
        sys.exit(1)


# Get the GDAL version dynamically
gdal_version = get_gdal_version()

# Setting up the package with dynamic GDAL version
setup(
    name="eomaji",
    version="0.1.0",
    description="A Python package for EOMaji",
    author="Marie Lund Larsen",
    author_email="mlla@dhigroup.com",
    license="Apache-2.0",
    packages=["workflows"],
    install_requires=[
        f"gdal=={gdal_version}",
        "numpy>=1.24.4",
        "scikit-learn>=1.5.2",
        "pyproj>=3.7.0",
        "scipy>=1.14.1",
        "openeo>=0.33.0",
        "numba==0.57.1",
        "pydms @ git+https://github.com/radosuav/pyDMS.git",
    ],
    extras_require={"dev": ["ipykernel>=6.29.5"]},
)

from typing import List
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display
from pystac_client import Client


def get_available_stac_dates(
    stac_url, collection_id, bbox, start_date, end_date, extra_query=None
):
    """
    Query a STAC endpoint using pystac-client and return a list of available acquisition dates.

    Args:
        stac_url (str): URL of the STAC API.
        collection_id (str): ID of the collection to query.
        bbox (list): [west, south, east, north] bounding box.
        start_date (str): ISO format date string (e.g. "2023-06-01").
        end_date (str): ISO format date string (e.g. "2023-06-30").
        extra_query (dict): Optional STAC query filters (e.g., {"cloud_cover": {"lt": 20}}).

    Returns:
        List of unique `datetime.date` objects.
    """
    catalog = Client.open(stac_url)

    search = catalog.search(
        collections=[collection_id],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query=extra_query,
        max_items=1000,
    )

    items = list(search.get_items())
    dates = sorted(
        {datetime.fromisoformat(item.datetime.isoformat()).date() for item in items}
    )

    return dates


def get_available_dates(
    start_date: str, end_date: str, bbox: List, max_cloud_cover: int = 20
):
    """
    Get available dates for Sentinel-2 and Sentinel-3 collections within a specified bounding box and date range.
    Returns a dropdown widget with the allowed dates.
    """

    sentinel_2_collection_id = "sentinel-2-l2a"
    sentinel_3_collection_id = "sentinel-3-sl-2-lst-ntc"
    stac_url = "https://stac.dataspace.copernicus.eu/v1"
    sentinel_2_dates = get_available_stac_dates(
        stac_url,
        sentinel_2_collection_id,
        bbox,
        start_date,
        end_date,
        extra_query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    sentinel_3_dates = get_available_stac_dates(
        stac_url, sentinel_3_collection_id, bbox, start_date, end_date
    )
    valid_dates = set(sentinel_2_dates) & set(sentinel_3_dates)

    dropdown = widgets.Dropdown(
        options=[
            (valid_date.strftime("%Y-%m-%d"), valid_date) for valid_date in valid_dates
        ],
        description="Pick a Date:",
    )
    display(dropdown)
    return dropdown

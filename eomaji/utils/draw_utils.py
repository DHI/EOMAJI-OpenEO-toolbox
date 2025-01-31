from ipyleaflet import Map, basemaps, basemap_to_tiles, DrawControl


def draw_aoi():
    # switch the base map here
    watercolor = basemap_to_tiles(basemaps.Esri.WorldStreetMap)
    m = Map(layers=(watercolor,), center=(40, 10), zoom=2)
    draw_control = DrawControl(
        polyline={},
        polygon={
            "shapeOptions": {"color": "#6bc2e5", "fillOpacity": 0.5}
        },  # Allow polygons
        circlemarker={},  # Disable circlemarker
        circle={},  # Disable circles
        rectangle={},  # Disable rectangles
    )
    polygons = []

    def handle_draw(self, action, geo_json):
        """Do something with the GeoJSON when it's drawn on the map"""
        # Print the GeoJSON
        polygons.append(geo_json["geometry"])

    draw_control.on_draw(handle_draw)

    m.add_control(draw_control)
    return m, polygons

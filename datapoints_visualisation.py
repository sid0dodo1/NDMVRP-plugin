from qgis.core import (
    QgsVectorLayer, QgsProject, QgsField, QgsFeature, QgsGeometry, QgsPointXY,
    QgsMarkerSymbol, QgsLineSymbol, QgsPalLayerSettings, QgsVectorLayerSimpleLabeling,
    QgsLayerTreeLayer, QgsLayerTreeGroup
)
from PyQt5.QtCore import QVariant
import csv
import re  # Import regular expression module
import pandas as pd

def visualize_points(df):
    # Fixed colors for different categories
    colors = ['#FF5733', '#33FF57', '#3366FF', '#FF33C8', '#FFFF33', '#33FFFF']

    def read_points_from_df(df):
        points = {}

        # Convert the DataFrame to a list of lists
        data = df.values.tolist()

        # Get the header
        header = list(df.columns)

        for row in data:
            id_type = re.match(r'^[A-Za-z]+', str(row[0])).group()  # Extract ID type from No

            if id_type not in points:
                points[id_type] = []

            points[id_type].append(row)

        return header, points

    header, points = read_points_from_df(df)

    # Create a group layer for each group and add it to the layer tree
    root = QgsProject.instance().layerTreeRoot()
    locations_group = QgsLayerTreeGroup("Locations")
    root.addChildNode(locations_group)

    for i, (id_type, group_points) in enumerate(points.items()):
        group_layer = QgsLayerTreeGroup(id_type)
        locations_group.addChildNode(group_layer)

        # Sort points based on IDs
        group_points.sort(key=lambda x: int(re.search(r'\d+$', x[0]).group()))  # Extract numeric part of IDs and sort based on that

        # Create line layer with dynamic fields
        line_fields = [QgsField(field_name, QVariant.String) for field_name in header]
        line_layer = QgsVectorLayer("LineString?crs=EPSG:4326", f"{id_type} Line", "memory")
        pr = line_layer.dataProvider()
        pr.addAttributes(line_fields)
        line_layer.updateFields()

        # Create a feature for the line
        feature = QgsFeature()

        # Set attributes for the feature
        feature.setAttributes(group_points[0])

        # Add the feature to the layer
        pr.addFeature(feature)

        # Set line symbol for the line layer
        symbol = QgsLineSymbol.createSimple({'color': colors[i], 'width': '1'})

        for j, row in enumerate(group_points):
            # Create point layer with dynamic fields
            point_fields = [QgsField(field_name, QVariant.String) for field_name in header]
            point_layer = QgsVectorLayer("Point?crs=EPSG:4326", f"{row[0]}", "memory")
            pr = point_layer.dataProvider()
            pr.addAttributes(point_fields)
            point_layer.updateFields()

            # Add the point and label
            point_feature = QgsFeature()
            point_feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(row[3]), float(row[2]))))
            point_feature.setAttributes(row)
            pr.addFeature(point_feature)
            point_layer.updateExtents()

            # Set marker symbol for the point layer
            symbol = QgsMarkerSymbol.createSimple({'color': colors[i], 'size': '3', 'outline_color': 'black', 'outline_width': '0.5'})  # Reduced size to '3'
            point_layer.renderer().setSymbol(symbol)

            # Enable labeling for the layer
            point_layer.setLabelsEnabled(True)

            # Configure the label settings
            label_settings = QgsPalLayerSettings()
            label_settings.fieldName = header[0]  # Assuming the first field in the header is the ID field
            label_settings.placement = QgsPalLayerSettings.AroundPoint
            point_layer.setLabeling(QgsVectorLayerSimpleLabeling(label_settings))

            # Add the point layer to the map and the group layer
            QgsProject.instance().addMapLayer(point_layer, False)
            group_layer.addLayer(point_layer)

    QgsProject.instance().write()
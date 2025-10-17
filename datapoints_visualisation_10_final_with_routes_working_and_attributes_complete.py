import os
import pandas as pd
import processing
from qgis.core import (
    QgsVectorLayer, QgsProject, QgsField, QgsFeature, QgsGeometry, QgsPointXY,
    QgsMarkerSymbol, QgsLineSymbol, QgsPalLayerSettings, QgsVectorLayerSimpleLabeling,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsFeatureRequest, QgsLayerTreeGroup, QgsLayerTreeLayer
)
from PyQt5.QtCore import QVariant
from PyQt5.QtGui import QColor
from random import randint

# Function to create a new vector layer
def create_vector_layer(layer_name):
    crs = QgsCoordinateReferenceSystem("EPSG:4326")
    layer = QgsVectorLayer("LineString?crs=epsg:4326", layer_name, "memory")
    
    provider = layer.dataProvider()
    provider.addAttributes([QgsField("vehicle_id", QVariant.String),
                            QgsField("starting_vertex", QVariant.String),
                            QgsField("ending_vertex", QVariant.String),
                            QgsField("loadcode_starting_vertex", QVariant.String),
                            QgsField("loadcode_ending_vertex", QVariant.String),
                            QgsField("starting_vertex_latitude", QVariant.Double),
                            QgsField("starting_vertex_longitude", QVariant.Double),
                            QgsField("ending_vertex_latitude", QVariant.Double),
                            QgsField("ending_vertex_longitude", QVariant.Double),
                            QgsField("Vehicle_StatusCode", QVariant.String),
                            QgsField("Time_Tuple", QVariant.String)])
    layer.updateFields()
    
    return layer

# Function to calculate shortest paths for a given road layer
def calculate_shortest_path(start_lat_lon, end_lat_lon, output_layer_name, road_layer):
    shortest_path_layer = create_vector_layer(output_layer_name)

    result = None

    try:
        result = processing.run("qgis:shortestpathpointtopoint",
                                {'INPUT': road_layer,
                                 'STRATEGY': 0,
                                 'START_POINT': start_lat_lon,
                                 'END_POINT': end_lat_lon,
                                 'PATH_TYPE': 0,
                                 'ENTRY_COST_CALCULATION': 0,
                                 'DIRECTION_FIELD': '',
                                 'VALUE_FORWARD': '',
                                 'VALUE_BACKWARD': '',
                                 'VALUE_BOTH': '',
                                 'DEFAULT_DIRECTION': 2,
                                 'SPEED_FIELD': '',
                                 'DEFAULT_SPEED': 50,
                                 'TOLERANCE': 0,
                                 'OUTPUT': 'TEMPORARY_OUTPUT'})
        travel_cost = result['TRAVEL_COST'] if result else None
    except Exception as e:
        print(f"Error calculating shortest path: {e}")
        travel_cost = None

    output_layer = result['OUTPUT'] if result else None
    
    if output_layer and output_layer.isValid():
        for feature in output_layer.getFeatures():
            shortest_path_layer.dataProvider().addFeatures([feature])

    return shortest_path_layer

# Function to get the plugin directory
def get_plugin_directory():
    try:
        # Assuming this script is part of a QGIS plugin
        return os.path.dirname(__file__)
    except NameError:
        # Fallback for environments where __file__ is not defined
        return os.getcwd()

# Function to get road layer path based on network compatibility
def get_road_layer_path(vehicle_network_compatibility):
    plugin_dir = get_plugin_directory()
    if vehicle_network_compatibility == 1:
        return os.path.join(plugin_dir, "Networks/Network_1.gpkg")
    elif vehicle_network_compatibility == 2:
        return os.path.join(plugin_dir, "Networks/Network_2.gpkg")
    elif vehicle_network_compatibility == 3:
        return os.path.join(plugin_dir, "Networks/Network_3.gpkg")
    else:
        return None

def safe_eval(cell):
    try:
        return eval(cell)
    except:
        return cell

def datapoints_visualisation_road_layer(points_data):

    # Create a group to store all layers
    group_name = "Output Layers"
    root = QgsProject.instance().layerTreeRoot()
    group = root.addGroup(group_name)

    # Create a dictionary to store unique colors for each Vehicle Unique Identity
    unique_colors = {}

    # Create a dictionary to store subgroups
    subgroups = {}

    # Loop through each row in the CSV file
    for _, row in points_data.iterrows():
        # Get the vehicle network compatibility directly
        vehicle_network_compatibility = safe_eval(row['Vehicle Network Compatibility'])

        road_layer_path = get_road_layer_path(vehicle_network_compatibility)

        if road_layer_path:
            road_layer = QgsVectorLayer(road_layer_path, "Road Layer", "ogr")
            if not road_layer.isValid():
                print("Road layer is not valid!")
                continue
        
            start_lat = row['starting vertex latitude']
            start_lon = row['starting vertex longitude']
            end_lat = row['ending vertex latitude']
            end_lon = row['ending vertex longitude']
        
            start_lat_lon = QgsPointXY(float(start_lon), float(start_lat))
            end_lat_lon = QgsPointXY(float(end_lon), float(end_lat))

            output_layer_name = f"Route from {row['starting vertex']} to {row['ending vertex']}"
            shortest_path_layer = calculate_shortest_path(start_lat_lon, end_lat_lon, output_layer_name, road_layer)

            if shortest_path_layer:
                # Add attributes to the attribute table
                shortest_path_layer.startEditing()
                for feature in shortest_path_layer.getFeatures():
                    feature['vehicle_id'] = str(row['Vehicle Unique Identity'])  # Ensure it's a string
                    feature['starting_vertex'] = row['starting vertex']
                    feature['ending_vertex'] = row['ending vertex']
                    feature['loadcode_starting_vertex'] = str(row['loadcode at starting vertex'])
                    feature['loadcode_ending_vertex'] = str(row['loadcode at ending vertex'])
                    feature['starting_vertex_latitude'] = float(row['starting vertex latitude'])
                    feature['starting_vertex_longitude'] = float(row['starting vertex longitude'])
                    feature['ending_vertex_latitude'] = float(row['ending vertex latitude'])
                    feature['ending_vertex_longitude'] = float(row['ending vertex longitude'])
                    feature['Vehicle_StatusCode'] = str(row['Vehicle StatusCode (just after the visit)'])
                    feature['Time_Tuple'] = str(row['Time Tuple'])
                    shortest_path_layer.updateFeature(feature)
                shortest_path_layer.commitChanges()

                # Assign unique color to the layer based on Vehicle Unique Identity
                vehicle_identity_str = str(row['Vehicle Unique Identity'])
                if vehicle_identity_str not in unique_colors:
                    unique_colors[vehicle_identity_str] = QColor.fromRgb(
                        randint(0, 255), randint(0, 255), randint(0, 255))
                color = unique_colors[vehicle_identity_str]
                symbol = QgsLineSymbol.createSimple({'color': f'{color.name()}', 'width': '0.8'})
                shortest_path_layer.renderer().setSymbol(symbol)

                # Add output layer to the subgroup
                if vehicle_identity_str not in subgroups:
                    subgroup = QgsLayerTreeGroup(vehicle_identity_str)
                    group.insertChildNode(0, subgroup)
                    subgroups[vehicle_identity_str] = subgroup
                subgroup = subgroups[vehicle_identity_str]
                subgroup.insertChildNode(0, QgsLayerTreeLayer(shortest_path_layer))

                # Add output layer to the map canvas
                QgsProject.instance().addMapLayer(shortest_path_layer, False)
        else:
            print(f"Unknown vehicle network compatibility: {vehicle_network_compatibility}. Cannot determine road layer path.")

    QgsProject.instance().write()  # Save changes to the project

# Example usage:
# points_data = pd.read_csv('path_to_your_input_csv.csv')
# datapoints_visualisation_road_layer(points_data)

import pandas as pd
from itertools import combinations
import processing
import os
from qgis.core import QgsVectorLayer, QgsProject, QgsWkbTypes, QgsCoordinateReferenceSystem, QgsField
from qgis.PyQt.QtCore import QVariant

# Function to create a new vector layer
def create_vector_layer():
    # Create the vector layer
    crs = QgsCoordinateReferenceSystem("EPSG:4326")
    layer_name = "Shortest Paths"
    layer = QgsVectorLayer("LineString?crs=epsg:4326", layer_name, "memory")
    
    # Add fields to the layer
    provider = layer.dataProvider()
    provider.addAttributes([QgsField("start_point_id", QVariant.String),
                            QgsField("end_point_id", QVariant.String),
                            QgsField("distance", QVariant.Double)])
    layer.updateFields()
    
    return layer

def shortestpath(points_data: pd.DataFrame, road_layer_path: str, output_csv_path: str, output_layer_name: str) -> pd.DataFrame:
    # Create a new vector layer for shortest paths
    shortest_paths_layer = create_vector_layer()
   
    # Get pairwise combinations of points
    point_combinations = combinations(points_data['No'], 2)

    # Create a list to store travel cost results
    travel_cost_results = []

    # Get the current directory
    current_directory = os.path.dirname(os.path.realpath(__file__))
    # Get the path of the file in the current directory
    file_path = os.path.join(current_directory, road_layer_path)

    # Iterate over point combinations
    for start_point_id, end_point_id in point_combinations:
        start_point = points_data[points_data['No'] == start_point_id].iloc[0]
        end_point = points_data[points_data['No'] == end_point_id].iloc[0]

        start_lat_lon = f"{start_point['Longitude']},{start_point['Latitude']} [EPSG:4326]"
        end_lat_lon = f"{end_point['Longitude']},{end_point['Latitude']} [EPSG:4326]"

        result = None  # Assign a default value to result

        try:
            # Run the processing algorithm
            result = processing.run("native:shortestpathpointtopoint", {
                'INPUT': file_path,
                'STRATEGY': 0,
                'DIRECTION_FIELD': '',
                'VALUE_FORWARD': '',
                'VALUE_BACKWARD': '',
                'VALUE_BOTH': '',
                'DEFAULT_DIRECTION': 2,
                'SPEED_FIELD': '',
                'DEFAULT_SPEED': 50,
                'TOLERANCE': 0,
                'START_POINT': start_lat_lon,
                'END_POINT': end_lat_lon,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })

            # Retrieve the travel cost from the result
            distance = result['TRAVEL_COST']

        except Exception as e:
            distance = None


        # Append the travel cost and point IDs to the list
        travel_cost_results.append({
            'start_point_id': start_point_id,
            'end_point_id': end_point_id,
            'distance': distance
        })

        try:
            # Run the processing algorithm for end to start
            result2 = processing.run("native:shortestpathpointtopoint", {
                'INPUT': file_path,
                'STRATEGY': 0,
                'DIRECTION_FIELD': '',
                'VALUE_FORWARD': '',
                'VALUE_BACKWARD': '',
                'VALUE_BOTH': '',
                'DEFAULT_DIRECTION': 2,
                'SPEED_FIELD': '',
                'DEFAULT_SPEED': 50,
                'TOLERANCE': 0,
                'START_POINT': end_lat_lon, # Reverse start and end points
                'END_POINT': start_lat_lon, # Reverse start and end points
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })

            # Retrieve the travel cost from the result
            travel_cost2 = result2['TRAVEL_COST']

        except Exception as e:
            travel_cost2 = None

        # Append the travel cost and point IDs to the list
        travel_cost_results.append({
            'start_point_id': end_point_id, # Reverse start and end points
            'end_point_id': start_point_id, # Reverse start and end points
            'distance': travel_cost2
        })

        # Retrieve the output vector layer
        output_layer = result['OUTPUT'] if result else None

        # Check if the output layer is valid and add features to the shortest paths layer
        if output_layer and output_layer.isValid():
            for feature in output_layer.getFeatures():
                shortest_paths_layer.addFeature(feature)

    # Convert the list of travel cost results to a DataFrame
    travel_cost_df = pd.DataFrame(travel_cost_results)

    # Save the travel cost results to a CSV file
    travel_cost_df.to_csv(output_csv_path, index=False)
    print(f"Travel cost results saved to {output_csv_path}")

    # Add the output CSV file as a dataset to QGIS
    output_layer = QgsVectorLayer(output_csv_path, output_layer_name, "ogr")
    if not output_layer.isValid():
        print("Failed to load the layer!")
    else:
        QgsProject.instance().addMapLayer(output_layer)
        print(f"{output_layer_name} added to QGIS!")

    return travel_cost_df

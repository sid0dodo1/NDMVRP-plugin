import os
import pandas as pd

# Helper function to safely evaluate strings into Python objects
def safe_eval(cell):
    try:
        return eval(cell)
    except:
        return cell

def output_breakdown(df, Locations_and_PickUp_Delivery_path):
    
    # Paths to the CSV files
    locations_file = os.path.join(Locations_and_PickUp_Delivery_path, "1 Locations and PickUp Delivery details.csv")
    vehicles_file = os.path.join(Locations_and_PickUp_Delivery_path, "0 Vehicles.csv")
    
    # Read the CSV files
    location_df = pd.read_csv(locations_file)
    vehicles_df = pd.read_csv(vehicles_file)
    
    # Create a dictionary mapping vehicle types to their network compatibility
    vehicle_compatibility = vehicles_df.set_index('Vehicle Type')['Vehicle Network Compatibility'].to_dict()
    
    # Initialize lists to store data for output CSV
    output_data = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        vehicle_identity = safe_eval(row['Vehicle Unique Identity'])
        vertices_visited = safe_eval(row['Vertices Visited'])
        loadcodes = safe_eval(row['LoadCodes at each Vertex'])
        statuscodes = safe_eval(row['Vehicle StatusCode (just after the visit)'])
        time_tuple = safe_eval(row['Time Tuple'])

        # Ensure statuscodes list is complete and has no empty entries
        statuscodes = [sc if sc is not None else 0 for sc in statuscodes]
    
        # Ensure time_tuple list is complete and has no empty entries
        time_tuple = [tt if tt is not None else 0 for tt in time_tuple]
        

        # Ensure the first entry of vertices_visited starts with the vehicle's unique identity
        vertices_visited.insert( 0, vehicle_identity[0] )
        loadcodes.insert(0, 0)  # Assuming starting loadcode as 0 for vehicle's starting point
        statuscodes.insert(0, 0)  # Assuming starting statuscode as 0 for vehicle's starting point
        time_tuple.insert(0, (0, 0, 0))  # Assuming starting time tuple as (0, 0, 0) for vehicle's starting point

        # Get the vehicle type from the vehicle identity
        vehicle_type = vehicle_identity[1].strip()
        
        # Get the vehicle network compatibility for the vehicle type
        network_compatibility = vehicle_compatibility.get(vehicle_type, None)

        # Generate ordered pairs and load codes
        for i in range(len(vertices_visited) - 1):
            starting_vertex = vertices_visited[i]
            ending_vertex = vertices_visited[i + 1]
        
            starting_loadcode = loadcodes[i] if loadcodes[i] is not None else 0
            ending_loadcode = loadcodes[i + 1] if i + 1 < len(loadcodes) and loadcodes[i + 1] is not None else 0
        
            # Extract latitude and longitude for starting vertex
            starting_vertex_data = location_df.loc[location_df['Sl. No.'] == starting_vertex]
            starting_vertex_latitude = starting_vertex_data['Latitude'].values[0] if not starting_vertex_data.empty else 0
            starting_vertex_longitude = starting_vertex_data['Longitude'].values[0] if not starting_vertex_data.empty else 0
        
            # Extract latitude and longitude for ending vertex
            ending_vertex_data = location_df.loc[location_df['Sl. No.'] == ending_vertex]
            ending_vertex_latitude = ending_vertex_data['Latitude'].values[0] if not ending_vertex_data.empty else 0
            ending_vertex_longitude = ending_vertex_data['Longitude'].values[0] if not ending_vertex_data.empty else 0
        
            # Append data to output list
            output_data.append({
                'Vehicle Unique Identity': vehicle_identity,
                'starting vertex': starting_vertex,
                'ending vertex': ending_vertex,
                'loadcode at starting vertex': starting_loadcode,
                'loadcode at ending vertex': ending_loadcode,
                'starting vertex latitude': starting_vertex_latitude,
                'starting vertex longitude': starting_vertex_longitude,
                'ending vertex latitude': ending_vertex_latitude,
                'ending vertex longitude': ending_vertex_longitude,
                'Vehicle StatusCode (just after the visit)': statuscodes[i + 1] if i + 1 < len(statuscodes) else 0,
                'Time Tuple': time_tuple[i + 1] if i + 1 < len(time_tuple) else (0, 0, 0),
                'Vehicle Network Compatibility': network_compatibility  # Add the network compatibility
            })

    # Create DataFrame from output data
    output_df = pd.DataFrame(output_data)
    return output_df

# Example usage
# input_file = 'D:/DataSet_RichVRP/OutPut (2).xlsx'
# df = pd.read_excel(input_file)

# Locations_and_PickUp_Delivery_path = 'D:/DataSet_RichVRP/'
# output_df = output_breakdown(df, Locations_and_PickUp_Delivery_path)

# output_file = 'D:/DataSet_RichVRP/OutPut ordered with latitude and longitude.csv'
# output_df.to_csv(output_file, index=False)
# print("Output CSV saved to", output_file)

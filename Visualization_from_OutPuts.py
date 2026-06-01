# How do I increase the cell width of the Jupyter version 7?
# https://stackoverflow.com/a/78487654/2879865

import os
import pandas as pd

current_path = os.getcwd()
print("\n Generating Vizualizations from Processed xlsx files; Current Path is: ", current_path)


# # <font color='red'>CSV File Reading starts</font>

# In[35]:

file_path = "userDefined_INPUT_location_of_CSV.txt"
with open(file_path, 'r') as file:
    current_Instance_CSV_location = file.read().strip()
    print(f"current_Instance_CSV_location path read from file: {current_Instance_CSV_location}")


file_path = "userDefined_OUTPUT_location.txt"
with open(file_path, 'r') as file:
    current_Instance_OUTPUT_Folder_Location = file.read().strip()
    print(f"current_Instance_OUTPUT_Folder_Location path read from file: {current_Instance_OUTPUT_Folder_Location}")


file_path = "visualization_file_name.txt"
with open(file_path, 'r') as file:
    visualization_fileName = file.read().strip() # This is name of the Process file (like "AllRouteSUM_OutPut_0.xlsx" or "MakeSPAN_OutPut_2.xlsx")
    print(f"visualization_fileName path read from file: {visualization_fileName}")


current_Vizualization_Processed_XLSX_Location = current_Instance_OUTPUT_Folder_Location + visualization_fileName



# Create a Folder within this OutPut Folder Location (for the respective Processed OutPut Visualization)

file_path = "visualization_FOLDER_name.txt"
with open(file_path, 'r') as file:
    current_visualization_Foldername = file.read().strip() # This is the name of the Folder where all the Route Outputs of a single Vizualization (i.e. Processed XLSX file) will be stored
    print(f"current_visualization_Foldername path read from file: {current_visualization_Foldername}")

current_Vizualization_Folder_Location = current_Instance_OUTPUT_Folder_Location + current_visualization_Foldername
os.makedirs(current_Vizualization_Folder_Location)





#OutPut_excel = pd.read_excel(default_location_of_OutPut+'OutPut_0.xlsx', index_col="Vehicle Unique Identity")
#OutPut_excel = pd.read_excel(default_location_of_OutPut+'MakeSPAN_OutPut_0.xlsx', index_col="Vehicle Unique Identity")
OutPut_excel = pd.read_excel(current_Vizualization_Processed_XLSX_Location, index_col="Vehicle Unique Identity")
# In[37]:


Path_of_Each_Vehicle = {}
# Step 3: Iterate over the rows of the DataFrame
for index, row in OutPut_excel.iterrows():
    print(f"\n Index: {index}")
    print(f"Key: {row['Vertices Visited']}")
    Path_of_Each_Vehicle[index] = row['Vertices Visited']
#     print(f"Value: {row['Value']}")

Path_of_Each_Vehicle


# In[38]:


# The ast.literal_eval function from Python's ast module safely evaluates strings containing Python literals
import ast

# Input dictionary with strings as keys and values
input_dict = Path_of_Each_Vehicle

# Initialize an empty dictionary to store the converted values
vehicle_routes = {}

# Iterate through the input dictionary
for key, value in Path_of_Each_Vehicle.items():
    # Convert the key and value from string to tuple and list respectively
    converted_key = ast.literal_eval(key)
    converted_value = ast.literal_eval(value)
    # Assign the converted key-value pair to the new dictionary
    vehicle_routes[converted_key] = converted_value

vehicle_routes


# In[39]:


# Appending the information of the starting Vehicle Depot in the vehicle_routes

for each_key in vehicle_routes:
    vehicle_routes[each_key].insert(0, each_key[0])
    
vehicle_routes


# In[40]:


csv_1_Locations_and_PickUp_Delivery_details = pd.read_csv(current_Instance_CSV_location+"1 Locations and PickUp Delivery details.csv", index_col = "Sl. No.")
#print("csv_1_Locations_and_PickUp_Delivery_details: ", csv_1_Locations_and_PickUp_Delivery_details)
# csv_1_Locations_and_PickUp_Delivery_details["Latitude"]["VD1"]
# csv_1_Locations_and_PickUp_Delivery_details["Longitude"]["VD1"]


# In[41]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

# # Example dictionary of vehicle routes using vertex names
# vehicle_routes = {
#     'Vehicle1': ['A', 'B', 'C', 'B', 'A'],
#     'Vehicle2': ['D', 'E', 'F', 'D'],
#     # Add more vehicles as needed
# }

# # Example dictionary of vertex coordinates (latitude, longitude)
# vertex_coords = {
#     'A': (0, 0),
#     'B': (1, 2),
#     'C': (3, 1),
#     'D': (2, 3),
#     'E': (4, 4),
#     'F': (5, 2)
# }


# In[42]:


# Example node types
node_types = dict(csv_1_Locations_and_PickUp_Delivery_details["Vertex Category"])

description = dict(csv_1_Locations_and_PickUp_Delivery_details["Description"])

remarks_comments = description = dict(csv_1_Locations_and_PickUp_Delivery_details["Remarks/Comments"])

# In[43]:


# node_symbols = {
#     'Depot': 's',
#     'Customer': 'o'
# }
node_symbols = {'Vehicle Depot' : 's',
                'WareHouse' : 'o',
                'Simultaneous Node' : 'p',
                'Transhipment Port' : '*',
                'Split Node' : '8',
                'Relief Centre' : 'P'}

node_colors = {'Vehicle Depot' : 'black',
                'WareHouse' : 'green',
                'Simultaneous Node' : 'blue',
                'Transhipment Port' : 'brown',
                'Split Node' : 'yellow',
                'Relief Centre' : 'red'}


# In[44]:


# for node, (x, y) in pos.items():
#     node_type = node_types.get(node)#, 'Customer')
#     print("node: ", node,";\t node_type: ",node_type)

#https://stackoverflow.com/questions/64276513/draw-dotted-or-dashed-rectangle-from-pil






import math


# In[45]:


# def plot_vehicle_route(vehicle, route, vertex_coords, node_types, node_symbols, ax):
def plot_vehicle_route(vehicle, route, ax, edgeColor):
    G = nx.DiGraph()
#     pos = {vertex: vertex_coords[vertex] for vertex in vertex_coords}
    pos = {vertex: (csv_1_Locations_and_PickUp_Delivery_details["Longitude"][vertex],csv_1_Locations_and_PickUp_Delivery_details["Latitude"][vertex])
           for vertex in dict(csv_1_Locations_and_PickUp_Delivery_details["Latitude"])}

    # Add nodes with coordinates
    for vertex in route:
        G.add_node(vertex)
            
            
    # Add edges and count the edges
    edge_count = {}
    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        if (u, v) in edge_count:
            edge_count[(u, v)] += 1
        else:
            edge_count[(u, v)] = 1
        
        if (v, u) in edge_count:
            edge_count[(v, u)] += 1
        else:
            edge_count[(v, u)] = 1
            
            
            
            
    # Draw edges with curved arrows
    drawn_edges = {}
    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        count = edge_count[(u, v)]
        total_count = drawn_edges.get((u, v), 0)
        total_count += 1
        drawn_edges[(u, v)] = total_count

        rad = (total_count - 1 - (count - 1) / 2) * 0.2
        
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            ax=ax, connectionstyle=f'arc3,rad={rad}',
            arrowstyle='-|>', arrowsize=15, edge_color=edgeColor
        )


    # Draw nodes with different symbols and colors
    drawn_labels = set()
    for node, (x, y) in pos.items():
        node_type = node_types.get(node)
        symbol = node_symbols[node_type]
        #ax.scatter(x, y, marker=symbol, s=100, label=node_type if node not in pos else "")
        color = node_colors[node_type]
        if node_type not in drawn_labels:
            ax.scatter(x, y, marker=symbol, s=100, color=color, label=node_type)
            drawn_labels.add(node_type)
        else:
            ax.scatter(x, y, marker=symbol, s=100, color=color)
        # Add node labels
        #ax.text(x, y, node, fontsize=12, ha='right')
        #ax.text(x, y, node, fontsize=12, ha='right', va='bottom') # This is fine as later edits will be done in EPS format in INKSCAPE
        
        #ax.text(x-0.25, y+0.25, node, fontsize=12, ha='right', va='bottom')
        ax.text(x-0.25, y+0.25, remarks_comments.get(node), fontsize=12, ha='right', va='bottom') # Choose the above or below to see the Node or Description as labels...
        #ax.text(x-0.25, y+0.25, description.get(node), fontsize=12, ha='right', va='bottom')
        
        #ax.text(x, y + 0.1, node, fontsize=12, ha='center', va='bottom')  # Adjusted position
        #ax.text(x, y + 0.25, node, fontsize=12, ha='center', va='bottom')  # Adjusted position

    
#     #ax.grid(False)  # Remove grid lines
#     ax.set_title(f'Route for {vehicle}')
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#     ax.legend()
    
    ax.set_title(f'Route for {vehicle}', fontsize=14)
#     ax.set_xlabel('Longitude', fontsize=13)
#     ax.set_ylabel('Latitude', fontsize=13)
    ax.set_xlabel(r'Longitude $\rightarrow$', fontsize=13)  # Add right arrow
    ax.set_ylabel(r'Latitude $\rightarrow$', fontsize=13)   # Add right arrow
    ax.legend()


# In[46]:


# # Set global font properties to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"





# import matplotlib as mpl

# # Set default font family
# mpl.rcParams['font.family'] = 'Times New Roman'

# # Ensure ticks are shown
# mpl.rcParams['xtick.labelsize'] = 10
# mpl.rcParams['ytick.labelsize'] = 10
# mpl.rcParams['axes.titlesize'] = 14
# mpl.rcParams['axes.labelsize'] = 13


# In[47]:


import random


import matplotlib.ticker as ticker

from matplotlib.ticker import MaxNLocator

# Plot each vehicle route
for vehicle, route in vehicle_routes.items():
    #fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(11 , 11))
    
    edgeColour = random.choice(['r', 'g', 'b', 'black'])
    print("Edge Colour is: ",edgeColour)
    
    #plot_vehicle_route(vehicle, route, vertex_coords, node_types, node_symbols, ax)
    plot_vehicle_route(vehicle, route, ax, edgeColour)
    
    
    
    

    # Add margins (padding) so that markers don't get clipped by the axes
    plt.margins(0.1)
    
    #plt.show()

    
    
    
    
    
    # Saving into different formats
#     fig.savefig(default_location_of_OutPut+'RoutePlan_'+str(vehicle)+'.png')
#     fig.savefig(default_location_of_OutPut+'RoutePlan_'+str(vehicle)+'.pdf')
#     fig.savefig(default_location_of_OutPut+'RoutePlan_'+str(vehicle)+'.tif')
#     fig.savefig(default_location_of_OutPut+'RoutePlan_'+str(vehicle)+'.tiff')
#     fig.savefig(default_location_of_OutPut+'RoutePlan_'+str(vehicle)+'.eps')
    fig.savefig(current_Vizualization_Folder_Location + f'Route_{vehicle}.png', bbox_inches='tight')
    fig.savefig(current_Vizualization_Folder_Location + f'Route_{vehicle}.pdf', bbox_inches='tight')
    fig.savefig(current_Vizualization_Folder_Location + f'Route_{vehicle}.tif', bbox_inches='tight')
    fig.savefig(current_Vizualization_Folder_Location + f'Route_{vehicle}.tiff', bbox_inches='tight')
    fig.savefig(current_Vizualization_Folder_Location + f'Route_{vehicle}.eps', format='eps', bbox_inches='tight')  # Save as EPS format

    plt.close(fig)  # Close the figure
    plt.close()
    plt.clf()       # Clear the current figure
    plt.cla()       # Clear the current axes

# In[ ]:





# ## The below code is entirely in matplotlib and the following parameters can be toggled for improving the visualization:
# 1) The factor in the shrink ratio (currently set to 1, increasing it will stop the arrow head from reaching its destination): shrink_ratio = 1 / arrow_length
# 
# 2) The mutation scale (to increase the arrowhead size): mutation_scale=15

# In[48]:




from matplotlib.path import Path
import numpy as np

def plot_vehicle_route(vehicle, route, ax, edgeColor):
    pos = {vertex: (csv_1_Locations_and_PickUp_Delivery_details["Longitude"][vertex], csv_1_Locations_and_PickUp_Delivery_details["Latitude"][vertex])
           for vertex in dict(csv_1_Locations_and_PickUp_Delivery_details["Latitude"])}

    # For scatterring only those Vertices which have been used for this vizualization
    subset_pos = {}

    # Draw edges with curved arrows
    edge_count = {}
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        if (u, v) in edge_count:
            edge_count[(u, v)] += 1
        else:
            edge_count[(u, v)] = 1

        if (v, u) in edge_count:
            edge_count[(v, u)] += 1
        else:
            edge_count[(v, u)] = 1

    drawn_edges = {}
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        count = edge_count[(u, v)]
        total_count = drawn_edges.get((u, v), 0)
        total_count += 1
        drawn_edges[(u, v)] = total_count

        rad = (total_count - 1 - (count - 1) / 2) * 0.2

        # Coordinates of start and end points
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        # Populating subset_pos
        subset_pos[u] = pos[u]
        subset_pos[v] = pos[v]


        # Adjust end position to stop slightly before the vertex icon
        dx, dy = x2 - x1, y2 - y1
        arrow_length = np.sqrt(dx**2 + dy**2)
        shrink_ratio = 1 / arrow_length
        x2_adj = x2 - dx * shrink_ratio
        y2_adj = y2 - dy * shrink_ratio

        # Create a curved path
        verts = [
            (x1, y1),  # Start point
            ((x1 + x2_adj) / 2, (y1 + y2_adj) / 2 + rad),  # Control point
            (x2_adj, y2_adj)  # End point
        ]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(verts, codes)

        # Create a PathPatch object
        # patch = patches.PathPatch(path, facecolor='none', edgecolor=edgeColor, lw=2)
        # ax.add_patch(patch)

        # Add arrowhead with larger size
        arrow = patches.FancyArrowPatch((x1, y1), (x2_adj, y2_adj), connectionstyle=f"arc3,rad={rad}",
                                        arrowstyle='-|>', mutation_scale=15, color=edgeColor)
        ax.add_patch(arrow)

    # Draw nodes with different symbols and colors
    drawn_labels = set()
    for node, (x, y) in subset_pos.items():
        node_type = node_types.get(node)
        symbol = node_symbols[node_type]
        color = node_colors[node_type]
        if node_type not in drawn_labels:
            ax.scatter(x, y, marker=symbol, s=100, color=color, label=node_type)
            drawn_labels.add(node_type)
        else:
            ax.scatter(x, y, marker=symbol, s=100, color=color)
        
        #ax.text(x - 0.25, y + 0.25, node, fontsize=12, ha='right', va='bottom')
        #ax.text(x - 0.25, y + 0.25, remarks_comments.get(node), fontsize=12, ha='right', va='bottom')
        ax.text(x - 0.25, y + 0.25, description.get(node), fontsize=12, ha='right', va='bottom') # Choose from either of the above to see the Remarks or the Node in the Inage labels
        

    ax.set_title(f'Route for {vehicle}', fontsize=14)
    ax.set_xlabel('Longitude', fontsize=13)
    ax.set_ylabel('Latitude', fontsize=13)
    ax.legend()

# Plot each vehicle route
for vehicle, route in vehicle_routes.items():
    fig, ax = plt.subplots(figsize=(11 , 11))

    edgeColour = random.choice(['r', 'g', 'b', 'black'])
    print("Edge Colour is: ", edgeColour)

    plot_vehicle_route(vehicle, route, ax, edgeColour)

    new_data = {"Longitude": dict(csv_1_Locations_and_PickUp_Delivery_details["Longitude"]),
                "Latitude": dict(csv_1_Locations_and_PickUp_Delivery_details["Latitude"])}

    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(labelsize=10)
    # ax.grid(True)

    #plt.show()

    # Saving into different formats
    fig.savefig(current_Vizualization_Folder_Location + f'Plan_{vehicle}.png', bbox_inches='tight')
    fig.savefig(current_Vizualization_Folder_Location + f'Plan_{vehicle}.pdf', bbox_inches='tight')
    fig.savefig(current_Vizualization_Folder_Location + f'Plan_{vehicle}.tif', bbox_inches='tight')
    fig.savefig(current_Vizualization_Folder_Location + f'Plan_{vehicle}.tiff', bbox_inches='tight')
    fig.savefig(current_Vizualization_Folder_Location + f'Plan_{vehicle}.eps', format='eps', bbox_inches='tight')

    plt.close(fig)  # Close the figure
    plt.close()
    plt.clf()       # Clear the current figure
    plt.cla()       # Clear the current axes


# In[ ]:





# In[ ]:





# ## The below Code will have the following options for the user to choose for the visualization (even after the routing heuristic runs, as this visulization file requires only the processed output files).
# 
# #### USER INPUT No.s:
# 1) Vehicle Route Legend as Rectangular Colour Boxes
# 
# 2) Vehicle Route Legend as line style with arrowhead (arrowhead does not always represent the actual image style)
# 
# 3) Vehicle Route Legend with only linestyle
# 
# ### To not display any legend, comment out the lines at the end of the code...

# In[79]:




from matplotlib.lines import Line2D

import copy


# In[85]:


drawn_labels = set()

def plot_vehicle_route(vehicle, route, ax, edgeColor, arrowStyle='-|>', lineStyle=None, lineWidth=None):
    
    pos = {vertex: (csv_1_Locations_and_PickUp_Delivery_details["Longitude"][vertex], csv_1_Locations_and_PickUp_Delivery_details["Latitude"][vertex])
           for vertex in dict(csv_1_Locations_and_PickUp_Delivery_details["Latitude"])}
    
    

    # Draw edges with curved arrows
    edge_count = {}
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        if (u, v) in edge_count:
            edge_count[(u, v)] += 1
        else:
            edge_count[(u, v)] = 1

        if (v, u) in edge_count:
            edge_count[(v, u)] += 1
        else:
            edge_count[(v, u)] = 1


            
            
            
            
            
            
            
    drawn_edges = {}
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        count = edge_count[(u, v)]
        total_count = drawn_edges.get((u, v), 0)
        total_count += 1
        drawn_edges[(u, v)] = total_count

        rad = (total_count - 1 - (count - 1) / 2) * 0.2

        # Coordinates of start and end points
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Adjust end position to stop slightly before the vertex icon
        dx, dy = x2 - x1, y2 - y1
        arrow_length = np.sqrt(dx**2 + dy**2)
        shrink_ratio = 1 / arrow_length
        x2_adj = x2 - dx * shrink_ratio
        y2_adj = y2 - dy * shrink_ratio

        # Create a curved path
        verts = [
            (x1, y1),  # Start point
            ((x1 + x2_adj) / 2, (y1 + y2_adj) / 2 + rad),  # Control point
            (x2_adj, y2_adj)  # End point
        ]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(verts, codes)

        

        
        # Add arrowhead with larger size
        arrow = patches.FancyArrowPatch((x1, y1), (x2_adj, y2_adj), connectionstyle=f"arc3,rad={rad}",
                                        arrowstyle=arrowStyle, mutation_scale=15, color=edgeColor, linestyle=lineStyle, linewidth=lineWidth)
        ax.add_patch(arrow)
        
        
        
           
        
        
        

    # Draw nodes with different symbols and colors
    global drawn_labels
    for node, (x, y) in pos.items():
        node_type = node_types.get(node)
        symbol = node_symbols[node_type]
        color = node_colors[node_type]
        if node_type not in drawn_labels:
            ax.scatter(x, y, marker=symbol, s=100, color=color, label=node_type)
            drawn_labels.add(node_type)
        else:
            ax.scatter(x, y, marker=symbol, s=100, color=color)
        
        ax.text(x - 0.25, y + 0.25, node, fontsize=12, ha='right', va='bottom') # Since this is a single large image of the enire problem's solution, it is best to use Node representations only and keep the below lines commented... # For the other labels, see similar scatter of the different images above...
        #ax.text(x - 0.25, y + 0.25, description.get(node), fontsize=12, ha='right', va='bottom')
        #ax.text(x - 0.25, y + 0.25, remarks_comments.get(node), fontsize=12, ha='right', va='bottom')

        
        
        
   
    


    
# Define arrow styles for each vehicle
#arrow_styles = ['-|>', '<|-', '-[', ']-']  # Please suggest only those arrow styles which retain the directional information (for example the arrowstyle of '<|-' alters the directional information)
arrow_styles = ['->', '-|>', ']->']  # Only directional arrow styles
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.ArrowStyle.html


# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
# line_styles = ['solid',
#                'dotted',
#                'dashed',
#                'dashdot',
#                'loosely dotted',
#                'dotted',
#                'densely dotted',
#                'long dash with offset',
#                'loosely dashed',
#                'dashed',
#                'densely dashed',
#                'loosely dashdotted',
#                'dashdotted',
#                'densely dashdotted',
#                'dashdotdotted',
#                'loosely dashdotdotted',
#                'densely dashdotdotted']
line_styles = ['solid', 'dotted', 'dashed', 'dashdot']


original_edgeColour = ['r', 'g', 'b', 'black']
dynamic_edgeColour = copy.deepcopy(original_edgeColour)


fig, ax = plt.subplots(figsize=(17 , 17))
legend_handles = []


# Plot all vehicle routes together using different colours
#for vehicle, route in vehicle_routes.items():
for i, (vehicle, route) in enumerate(vehicle_routes.items()):
    
    def choose_edge_colour_circularly(): # This function ensured there is no repeat in colours in the vehicular routes...
        global dynamic_edgeColour
        if len(dynamic_edgeColour)==0:
            dynamic_edgeColour = copy.deepcopy(original_edgeColour)
        edgeColour = random.choice(dynamic_edgeColour)
        print("Edge Colour is: ", edgeColour)
        dynamic_edgeColour.remove(edgeColour)
        return edgeColour
    edgeColour = choose_edge_colour_circularly()
    
    arrowStyle = arrow_styles[i % len(arrow_styles)]
    lineStyle = line_styles[i % len(line_styles)]
    
    # Here I want to add the chosen coloured edge as a legend with the Vehicles' name in the legend text
    
    #plot_vehicle_route(vehicle, route, ax, edgeColour)
    plot_vehicle_route(vehicle, route, ax, edgeColour, arrowStyle, lineStyle)
    
    
    
    
    
    
    # Add custom legend entry for each vehicle (I also suggested ChatGPT to provide how to differ the arrow styles for each vehicle as well):
    # USER INPUT 1: The following code shows a rectangular legend box having the same colour of the route...
    #legend_handles.append(patches.FancyArrowPatch((0, 0), (1, 0), connectionstyle="arc3,rad=0", arrowstyle=arrowStyle, linestyle=lineStyles, mutation_scale=15, color=edgeColour, label=vehicle))
    # USER INPUT 2: The following line appends the type of EDGE in the legend alongwith an arrowHead (which may not correspond directly to the arrowhead chosen at random)
    #legend_handles.append(Line2D([0], [0], color=edgeColour, linestyle=lineStyle, linewidth=2, label=vehicle, marker=arrowStyle[-1] if arrowStyle.startswith('-') else None, markersize=10, markeredgecolor=edgeColour, markerfacecolor=edgeColour))
    # USER INPUT 3: The following line appends only the type of EDGE in the legend
    legend_handles.append(Line2D([0], [1], color=edgeColour, linestyle=lineStyle, linewidth=2, label=vehicle))
    
    
    

# new_data = {"Longitude": dict(csv_1_Locations_and_PickUp_Delivery_details["Longitude"]),
#             "Latitude": dict(csv_1_Locations_and_PickUp_Delivery_details["Latitude"])}

    
    

 
    
    
    
ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
ax.yaxis.set_major_locator(MaxNLocator(nbins=15))
ax.tick_params(labelsize=10)

# ax.grid(True)
    
ax.set_title(f"Representation of all Vehicles' Routes", fontsize=14)
ax.set_xlabel('Longitude', fontsize=13)
ax.set_ylabel('Latitude', fontsize=13)


#ax.legend()


# Padding: So that markers don't get clipped by the axes; as well as the Node Legends don't overlap over the routes...
plt.margins(0.05 + random.random()/5)
# plt.margins(random.random()/4)




# For Node Type Legend
if random.random() > 0.5 :
    node_legend = ax.legend(loc='upper left', fontsize=11, title="Vertex Categories:\n", title_fontsize = 12)
else:
    node_legend = ax.legend(loc='upper right', fontsize=11, title="Vertex Categories:\n", title_fontsize = 12)
#node_legend = ax.legend(loc='upper left', fontsize=12, title='Node Types') # Default fontsize is 10
ax.add_artist(node_legend) # This line seems not necessary as the legend is displayed directly... # But, if both legends need to be visualized





# For Vehicle Route legend
#vehicle_legend = ax.legend(handles=legend_handles, loc='lower right', fontsize=12, title='Vehicle Routes')
if random.random() > 0.5 :
    ax.legend(handles=legend_handles, loc='lower right', fontsize=10, title='Vehicle Routes:~\n', title_fontsize = 12)
else:
    ax.legend(handles=legend_handles, loc='lower left', fontsize=10, title='Vehicle Routes:~\n', title_fontsize = 12)

    
    
    
#https://stackoverflow.com/questions/13018115/matplotlib-savefig-image-size-with-bbox-inches-tight


    
# Show the plot
#plt.show()

# Saving into different formats
fig.savefig(current_Vizualization_Folder_Location + f'Combined.png', bbox_inches='tight')
fig.savefig(current_Vizualization_Folder_Location + f'Combined.pdf', bbox_inches='tight')
fig.savefig(current_Vizualization_Folder_Location + f'Combined.tif', bbox_inches='tight')
fig.savefig(current_Vizualization_Folder_Location + f'Combined.tiff', bbox_inches='tight')
fig.savefig(current_Vizualization_Folder_Location + f'Combined.eps', format='eps', bbox_inches='tight')

plt.close(fig)  # Close the figure
plt.close()
plt.clf()       # Clear the current figure
plt.cla()       # Clear the current axes

# # # Please refer to QGIS for the actual routes...

# input("Wait for enter 8")

# ## Monitor your memory usage to better understand when and where memory usage spikes.
# import psutil

# input("Wait for enter 5") 
# process = psutil.Process(os.getpid())
# input("Wait for enter 3") 
# print("\n MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM")
# print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
# input("Wait for enter 3") 
# print("MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM \n")
# input("Wait for enter 3") 




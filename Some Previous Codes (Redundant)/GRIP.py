#!/usr/bin/env python
# coding: utf-8


get_ipython().run_line_magic('autosave', '11')
import time


import os
current_path = os.getcwd()
print(current_path)
print("GRIP: Greedy Route Insertion and Perturbation")


# from google.colab import drive
# drive.mount('/content/drive')
# _0_Cargo=pd.read_excel("/content/drive/MyDrive/Original_dataset_for_objective2 - Copy/Basic_Instances/Instance2/0 Cargo.xlsx",sheet_name=None,index_col=0)
# _0_Vehicles=pd.read_excel("/content/drive/MyDrive/Original_dataset_for_objective2 - Copy/Basic_Instances/Instance2/0 Vehicles.xlsx",sheet_name=None,index_col=0)
# _1_Locations_and_PickUp_Delivery_details=pd.read_excel("/content/drive/MyDrive/Original_dataset_for_objective2 - Copy/Basic_Instances/Instance2/1 Locations and PickUp Delivery details.xlsx",sheet_name=None,index_col=0)
# _1_Vehicle_Compatibility=pd.read_excel("/content/drive/MyDrive/Original_dataset_for_objective2 - Copy/Basic_Instances/Instance2/1 Vehicle Compatibility.xlsx",sheet_name=None,index_col=0)
# print(_1_Locations_and_PickUp_Delivery_details['Vehicle Depots'])



import pandas as pd

_0_Cargo=pd.read_excel("0 Cargo.xlsx",sheet_name=None,index_col=0)
_0_Vehicles=pd.read_excel("0 Vehicles.xlsx",sheet_name=None,index_col=0)
_1_Locations_and_PickUp_Delivery_details=pd.read_excel("1 Locations and PickUp Delivery details.xlsx",sheet_name=None,index_col=0)
_1_Vehicle_Compatibility=pd.read_excel("1 Vehicle Compatibility.xlsx",sheet_name=None,index_col=0)



Cargo_Delivery_Type_arr=_0_Cargo["Delivery"].index.tolist()
Cargo_Delivery_Description_dict=_0_Cargo["Delivery"]["Description"]
Cargo_Delivery_UnitWeight_dict=_0_Cargo["Delivery"]["Unit Weight"]
Cargo_Delivery_UnitVolume_dict=_0_Cargo["Delivery"]["Unit Volume"]

#print(Cargo_Delivery_UnitWeight_dict['CC1D'])

Cargo_PickUp_Type_arr=_0_Cargo["PickUp"].index.tolist()
Cargo_PickUp_Description_dict=_0_Cargo["PickUp"]["Description"]
Cargo_PickUp_UnitWeight_dict=_0_Cargo["PickUp"]["Unit Weight"]
Cargo_PickUp_UnitVolume_dict=_0_Cargo["PickUp"]["Unit Volume"]


Vehicles_Specifications_VehicleType_arr=_0_Vehicles["Specifications"].index.tolist()
#print(Vehicles_Specifications_VehicleType_arr)
Vehicles_Specifications_Description_dict=_0_Vehicles["Specifications"]["Description"]
Vehicles_Specifications_WeightCapacity_dict=_0_Vehicles["Specifications"]["Weight Capacity"]
Vehicles_Specifications_VolumeCapacity_dict=_0_Vehicles["Specifications"]["Volume Capacity"]
Vehicles_Specifications_OpenTour_dict=_0_Vehicles["Specifications"]["Need for return to the starting Depot"]
Vehicles_Specifications_VehicleNetworkCompatibility_dict=_0_Vehicles["Specifications"]["Vehicle Network Compatibility (OSM)"]
#print(Vehicles_Specifications_OpenTour_dict)


LocationPickupDelivery_VehicleDepots_SlNo_arr=_1_Locations_and_PickUp_Delivery_details["Vehicle Depots"].index.tolist()
LocationPickupDelivery_VehicleDepots_Description_dict=_1_Locations_and_PickUp_Delivery_details["Vehicle Depots"]["Description"]
LocationPickupDelivery_VehicleDepots_Latitude_dict=_1_Locations_and_PickUp_Delivery_details["Vehicle Depots"]["Latitude"]
LocationPickupDelivery_VehicleDepots_Longitude_dict=_1_Locations_and_PickUp_Delivery_details["Vehicle Depots"]["Longitude"]
#print(LocationPickupDelivery_VehicleDepots_SlNo_arr)
#print(LocationPickupDelivery_VehicleDepots_Longitude_dict["VD1"])
LocationPickupDelivery_VehicleDepots_VehicleTypesAvailable_dict={}
for Vehicle_Type in Vehicles_Specifications_VehicleType_arr:
    LocationPickupDelivery_VehicleDepots_VehicleTypesAvailable_dict[Vehicle_Type]=_1_Locations_and_PickUp_Delivery_details["Vehicle Depots"][Vehicle_Type]
#print(LocationPickupDelivery_VehicleDepots_VehicleTypesAvailable_dict["VT1"]["VD1"])
#print(LocationPickupDelivery_VehicleDepots_VehicleTypesAvailable_dict["VT4"])
#print(LocationPickupDelivery_VehicleDepots_VehicleTypes_dict["VT4"]["VD3"])
#print(type(LocationPickupDelivery_VehicleDepots_VehicleTypesAvailable_dict["VT2"]["VD3"]))
#print(type(LocationPickupDelivery_VehicleDepots_Longitude_dict["VD1"]))
#print(type(LocationPickupDelivery_VehicleDepots_Longitude_dict))
#print(_1_Locations_and_PickUp_Delivery_details["Vehicle Depots"]["VT1"])



LocationPickupDelivery_Warehouses_SlNo_arr=_1_Locations_and_PickUp_Delivery_details["WareHouses"].index.tolist()
LocationPickupDelivery_Warehouses_Description_dict=_1_Locations_and_PickUp_Delivery_details["WareHouses"]["Description"]
LocationPickupDelivery_Warehouses_Latitude_dict=_1_Locations_and_PickUp_Delivery_details["WareHouses"]["Latitude"]
LocationPickupDelivery_Warehouses_Longitude_dict=_1_Locations_and_PickUp_Delivery_details["WareHouses"]["Longitude"]
#print(LocationPickupDelivery_Warehouses_SlNo_arr)
#print(LocationPickupDelivery_Warehouses_Description_dict)
#print(LocationPickupDelivery_Warehouses_Latitude_dict)
#print(LocationPickupDelivery_Warehouses_Longitude_dict)
LocationPickupDelivery_Warehouses_DeliveryCargoAvailable_dict={}
for CompatibleCargo_Delivery_Type in Cargo_Delivery_Type_arr:
    LocationPickupDelivery_Warehouses_DeliveryCargoAvailable_dict[CompatibleCargo_Delivery_Type]=_1_Locations_and_PickUp_Delivery_details["WareHouses"][CompatibleCargo_Delivery_Type]
#print(LocationPickupDelivery_Warehouses_DeliveryCargoAvailable_dict["CC5D"])



LocationPickupDelivery_SimultaneousNodes_SlNo_arr=_1_Locations_and_PickUp_Delivery_details["Simultaneous Nodes"].index.tolist()
LocationPickupDelivery_SimultaneousNodes_Description_dict=_1_Locations_and_PickUp_Delivery_details["Simultaneous Nodes"]["Description"]
LocationPickupDelivery_SimultaneousNodes_Latitude_dict=_1_Locations_and_PickUp_Delivery_details["Simultaneous Nodes"]["Latitude"]
LocationPickupDelivery_SimultaneousNodes_Longitude_dict=_1_Locations_and_PickUp_Delivery_details["Simultaneous Nodes"]["Longitude"]
# print(LocationPickupDelivery_SimultaneousNodes_Longitude_dict)
LocationPickupDelivery_SimultaneousNodes_DeliveryCargoRequired_dict={}
for CompatibleCargo_Delivery_Type in Cargo_Delivery_Type_arr:
    LocationPickupDelivery_SimultaneousNodes_DeliveryCargoRequired_dict[CompatibleCargo_Delivery_Type]=_1_Locations_and_PickUp_Delivery_details["Simultaneous Nodes"][CompatibleCargo_Delivery_Type]
# print(LocationPickupDelivery_SimultaneousNodes_DeliveryCargoRequired_dict["CC3D"])
LocationPickupDelivery_SimultaneousNodes_PickUpCargoAwaiting_dict={}
for CompatibleCargo_PickUp_Type in Cargo_PickUp_Type_arr:
    LocationPickupDelivery_SimultaneousNodes_PickUpCargoAwaiting_dict[CompatibleCargo_PickUp_Type]=_1_Locations_and_PickUp_Delivery_details["Simultaneous Nodes"][CompatibleCargo_PickUp_Type]
# print(LocationPickupDelivery_SimultaneousNodes_PickUpCargoAwaiting_dict["CC3P"])



LocationPickupDelivery_TranshipmentPorts_SlNo_arr=_1_Locations_and_PickUp_Delivery_details["Transhipment Ports"].index.tolist()
LocationPickupDelivery_TranshipmentPorts_Description_dict=_1_Locations_and_PickUp_Delivery_details["Transhipment Ports"]["Description"]
LocationPickupDelivery_TranshipmentPorts_Latitude_dict=_1_Locations_and_PickUp_Delivery_details["Transhipment Ports"]["Latitude"]
LocationPickupDelivery_TranshipmentPorts_Longitude_dict=_1_Locations_and_PickUp_Delivery_details["Transhipment Ports"]["Longitude"]
# print(LocationPickupDelivery_TranshipmentPorts_Longitude_dict)
LocationPickupDelivery_TranshipmentPorts_DeliveryCargoCompatible_dict={}
for CompatibleCargo_Delivery_Type in Cargo_Delivery_Type_arr:
    LocationPickupDelivery_TranshipmentPorts_DeliveryCargoCompatible_dict[CompatibleCargo_Delivery_Type]=_1_Locations_and_PickUp_Delivery_details["Transhipment Ports"][CompatibleCargo_Delivery_Type]
#print(LocationPickupDelivery_TranshipmentPorts_DeliveryCargoCompatible_dict["CC3D"])
LocationPickupDelivery_TranshipmentPorts_PickUpCargoCompatible_dict={}
for CompatibleCargo_PickUp_Type in Cargo_PickUp_Type_arr:
    LocationPickupDelivery_TranshipmentPorts_PickUpCargoCompatible_dict[CompatibleCargo_PickUp_Type]=_1_Locations_and_PickUp_Delivery_details["Transhipment Ports"][CompatibleCargo_PickUp_Type]
# print(LocationPickupDelivery_TranshipmentPorts_PickUpCargoCompatible_dict["CC3P"])



LocationPickupDelivery_SplitNodes_SlNo_arr=_1_Locations_and_PickUp_Delivery_details["Split Nodes"].index.tolist()
LocationPickupDelivery_SplitNodes_Description_dict=_1_Locations_and_PickUp_Delivery_details["Split Nodes"]["Description"]
LocationPickupDelivery_SplitNodes_Latitude_dict=_1_Locations_and_PickUp_Delivery_details["Split Nodes"]["Latitude"]
LocationPickupDelivery_SplitNodes_Longitude_dict=_1_Locations_and_PickUp_Delivery_details["Split Nodes"]["Longitude"]
# print(LocationPickupDelivery_SplitNodes_Longitude_dict)
LocationPickupDelivery_SplitNodes_DeliveryCargoRequired_dict={}
for CompatibleCargo_Delivery_Type in Cargo_Delivery_Type_arr:
    LocationPickupDelivery_SplitNodes_DeliveryCargoRequired_dict[CompatibleCargo_Delivery_Type]=_1_Locations_and_PickUp_Delivery_details["Split Nodes"][CompatibleCargo_Delivery_Type]
# print(LocationPickupDelivery_SplitNodes_DeliveryCargoRequired_dict["CC3D"])
LocationPickupDelivery_SplitNodes_PickUpCargoAwaiting_dict={}
for CompatibleCargo_PickUp_Type in Cargo_PickUp_Type_arr:
    LocationPickupDelivery_SplitNodes_PickUpCargoAwaiting_dict[CompatibleCargo_PickUp_Type]=_1_Locations_and_PickUp_Delivery_details["Split Nodes"][CompatibleCargo_PickUp_Type]
# print(LocationPickupDelivery_SplitNodes_PickUpCargoAwaiting_dict["CC3P"])



LocationPickupDelivery_ReliefCentres_SlNo_arr=_1_Locations_and_PickUp_Delivery_details["Relief Centres"].index.tolist()
LocationPickupDelivery_ReliefCentres_Description_dict=_1_Locations_and_PickUp_Delivery_details["Relief Centres"]["Description"]
LocationPickupDelivery_ReliefCentres_Latitude_dict=_1_Locations_and_PickUp_Delivery_details["Relief Centres"]["Latitude"]
LocationPickupDelivery_ReliefCentres_Longitude_dict=_1_Locations_and_PickUp_Delivery_details["Relief Centres"]["Longitude"]
# print(LocationPickupDelivery_ReliefCentres_Longitude_dict)
LocationPickupDelivery_ReliefCentres_PickUpCargoSpaceAvailable_dict={}
for CompatibleCargo_PickUp_Type in Cargo_PickUp_Type_arr:
    LocationPickupDelivery_ReliefCentres_PickUpCargoSpaceAvailable_dict[CompatibleCargo_PickUp_Type]=_1_Locations_and_PickUp_Delivery_details["Relief Centres"][CompatibleCargo_PickUp_Type]
# print(LocationPickupDelivery_ReliefCentres_PickUpCargoSpaceAvailable_dict["CC3P"])


VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict={}
for CompatibleCargo_Delivery_Type in Cargo_Delivery_Type_arr:
    VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[CompatibleCargo_Delivery_Type]=_1_Vehicle_Compatibility["Loading Unloading Time"][CompatibleCargo_Delivery_Type]
#print(VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict["CC5D"])
#print(VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict["CC2D"],"\n")
#print("\n",VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict["CC2D"]["VT1"],"\n")

VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict={}
for CompatibleCargo_PickUp_Type in Cargo_PickUp_Type_arr:
    VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[CompatibleCargo_PickUp_Type]=_1_Vehicle_Compatibility["Loading Unloading Time"][CompatibleCargo_PickUp_Type]
#print(VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict["CC3P"]["VT4"])
#print("\n",VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict["CC2P"]["VT1"])



# Table 1: Sets and Parameters
VD=LocationPickupDelivery_VehicleDepots_SlNo_arr
W=LocationPickupDelivery_Warehouses_SlNo_arr
NM=LocationPickupDelivery_SimultaneousNodes_SlNo_arr
NP=LocationPickupDelivery_SplitNodes_SlNo_arr
N=NM+NP
TP=LocationPickupDelivery_TranshipmentPorts_SlNo_arr
RC=LocationPickupDelivery_ReliefCentres_SlNo_arr
V_0=W+N+TP+RC
#V=VD+V_0 #This is possibly never used as certain connections here are not present
#E=
#G=

#v= #All vehicles
vT={}
vN={}
for h in LocationPickupDelivery_VehicleDepots_SlNo_arr:
    vT[h]=[]
    for k in Vehicles_Specifications_VehicleType_arr:
        #print(k,h)
        #print(_1_Locations_and_PickUp_Delivery_details["Vehicle Depots"][k][h])
        if _1_Locations_and_PickUp_Delivery_details["Vehicle Depots"][k][h]:
            vT[h].append(k)
            vN[h,k]=range(1,_1_Locations_and_PickUp_Delivery_details["Vehicle Depots"][k][h]+1)
            vN[h,k]=_1_Locations_and_PickUp_Delivery_details["Vehicle Depots"][k][h]
print(vT)
print(vN)



# print("This l_max initialization needs to be updated since currently we are using only 3 layers")
# l_max={} # Assume current initial levels for all vehicles to be same and equal to 3. For less than this value, some equations would become incorrect/redundant
# for h in LocationPickupDelivery_VehicleDepots_SlNo_arr:
#     for k in vT[h]:
#         for vehicle_number in vN[h,k]:
#             v=(h,k,vehicle_number)
#             l_max[v]=5
#             print(v," : ",l_max[v])
# #print("Vehicle : Levels available")


DY=Cargo_Delivery_Type_arr
PU=Cargo_PickUp_Type_arr
#print(PU)

CC={}
#print("\n",VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict["CC2D"]["VT1"],"\n")
for vehicle_types in Vehicles_Specifications_VehicleType_arr:
    CC[vehicle_types]=[]
    for pickup_cargos in PU:
        #print(pickup_cargos , vehicle_types, VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[pickup_cargos][vehicle_types])
        if VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[pickup_cargos][vehicle_types] > 0:
            print(pickup_cargos , vehicle_types, VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[pickup_cargos][vehicle_types])
            CC[vehicle_types].append(pickup_cargos)
    for delivery_cargos in DY:
        #print(delivery_cargos, vehicle_types, VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[delivery_cargos][vehicle_types])
        if VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[delivery_cargos][vehicle_types] > 0:
            print(delivery_cargos, vehicle_types, VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[delivery_cargos][vehicle_types])
            CC[vehicle_types].append(delivery_cargos)
#print("\n",CC)

Q={}
#print(LocationPickupDelivery_Warehouses_DeliveryCargoAvailable_dict[CompatibleCargo_Delivery_Type])

for warehouses in W: # each DY in W
    for delivery_cargos in DY:
        Q[delivery_cargos,warehouses]=LocationPickupDelivery_Warehouses_DeliveryCargoAvailable_dict[delivery_cargos][warehouses]
        #print(warehouses, delivery_cargos," : ",Q[warehouses,delivery_cargos])

for simultaneous_nodes in NM:
    for pickup_cargos in PU:
        Q[pickup_cargos,simultaneous_nodes]=LocationPickupDelivery_SimultaneousNodes_PickUpCargoAwaiting_dict[pickup_cargos][simultaneous_nodes]
        #print(simultaneous_nodes,pickup_cargos," : ",Q[simultaneous_nodes,pickup_cargos])
    for delivery_cargos in DY:
        Q[delivery_cargos,simultaneous_nodes]=LocationPickupDelivery_SimultaneousNodes_DeliveryCargoRequired_dict[delivery_cargos][simultaneous_nodes]
        #print(simultaneous_nodes, delivery_cargos," : ",Q[simultaneous_nodes,delivery_cargos])
# each CC in N
for split_nodes in NP:
    for pickup_cargos in PU:
        Q[pickup_cargos,split_nodes]=LocationPickupDelivery_SplitNodes_PickUpCargoAwaiting_dict[pickup_cargos][split_nodes]
        #print(split_nodes,pickup_cargos," : ",Q[split_nodes,pickup_cargos])
    for delivery_cargos in DY:
        Q[delivery_cargos,split_nodes]=LocationPickupDelivery_SplitNodes_DeliveryCargoRequired_dict[delivery_cargos][split_nodes]
        #print(split_nodes, delivery_cargos," : ",Q[split_nodes,delivery_cargos])

for relief_centres in RC: # each PU in RC
    for pickup_cargos in PU:
        Q[pickup_cargos,relief_centres]=LocationPickupDelivery_ReliefCentres_PickUpCargoSpaceAvailable_dict[pickup_cargos][relief_centres]
        #print(relief_centres,pickup_cargos," : ",Q[relief_centres,pickup_cargos])

print(Q)


CP={}

for transhipment_ports in TP:
    CP[transhipment_ports]=[]
    for pickup_cargos in PU:
        if LocationPickupDelivery_TranshipmentPorts_PickUpCargoCompatible_dict[pickup_cargos][transhipment_ports]:
            CP[transhipment_ports].append(pickup_cargos)
    for delivery_cargos in DY:
        if LocationPickupDelivery_TranshipmentPorts_DeliveryCargoCompatible_dict[delivery_cargos][transhipment_ports]:
            CP[transhipment_ports].append(delivery_cargos)

#print(CP)


All_Longitudes = {**LocationPickupDelivery_VehicleDepots_Longitude_dict,**LocationPickupDelivery_Warehouses_Longitude_dict,**LocationPickupDelivery_SimultaneousNodes_Longitude_dict,**LocationPickupDelivery_TranshipmentPorts_Longitude_dict,**LocationPickupDelivery_SplitNodes_Longitude_dict,**LocationPickupDelivery_ReliefCentres_Longitude_dict}
All_Latitudes = {**LocationPickupDelivery_VehicleDepots_Latitude_dict,**LocationPickupDelivery_Warehouses_Latitude_dict,**LocationPickupDelivery_SimultaneousNodes_Latitude_dict,**LocationPickupDelivery_TranshipmentPorts_Latitude_dict,**LocationPickupDelivery_SplitNodes_Latitude_dict,**LocationPickupDelivery_ReliefCentres_Latitude_dict}

C={}
for vehicle_types in Vehicles_Specifications_VehicleType_arr:
    p = float(_0_Vehicles["Specifications"]["p-value used for generating Test Cases"][vehicle_types])
    #print(p," : ",type(p))
    for source in All_Longitudes:
        for sink in All_Latitudes:
            
            lat_diff = abs(All_Latitudes[source]-All_Latitudes[sink])
            lon_diff = abs(All_Longitudes[source]-All_Longitudes[sink])
            C[source,sink,vehicle_types] = ( (lat_diff**p + lon_diff**p)**(1/p) )/1000 #Converting distance from meter to kilometers
            #print(source,sink,vehicle_types," : ",C[source,sink,vehicle_types])
#print(C)
#print(len(C))


C



M = 9999999
U={}
CV={}
for pickup_cargos in PU:
    CV[pickup_cargos]=[]
    for vehicle_types in Vehicles_Specifications_VehicleType_arr:
        #print(pickup_cargos , vehicle_types, VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[pickup_cargos][vehicle_types])
        if VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[pickup_cargos][vehicle_types] > 0:
            #print(pickup_cargos , vehicle_types, VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[pickup_cargos][vehicle_types])
            U[vehicle_types,pickup_cargos]=VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[pickup_cargos][vehicle_types]
            CV[pickup_cargos].append(vehicle_types)
for delivery_cargos in DY:
    CV[delivery_cargos]=[]
    for vehicle_types in Vehicles_Specifications_VehicleType_arr:
        #print(delivery_cargos, vehicle_types, VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[delivery_cargos][vehicle_types])
        if VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[delivery_cargos][vehicle_types] > 0:
            #print(delivery_cargos, vehicle_types, VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[delivery_cargos][vehicle_types])
            U[vehicle_types,delivery_cargos]=VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[delivery_cargos][vehicle_types]
            CV[delivery_cargos].append(vehicle_types)
print(U)
print(CV)


# for cargos in VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict:
#     for vehicle_types in VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[cargos]:
#         print(vehicle_types)
#         if vehicle_types:
#             print(vehicle_types)
#         print()
#     #CC[k]=

# for k in VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict:
#     print(k)
#     #CC[k]=



E={}
for vehicle_types in Vehicles_Specifications_VehicleType_arr:
    #print(Vehicles_Specifications_VolumeCapacity_dict[vehicle_types])
    E[vehicle_types]=Vehicles_Specifications_VolumeCapacity_dict[vehicle_types]
for pickup_cargos in PU:
    E[pickup_cargos]=Cargo_PickUp_UnitVolume_dict[pickup_cargos]
for delivery_cargos in DY:
    E[delivery_cargos]=Cargo_Delivery_UnitVolume_dict[delivery_cargos]
#print(E)

#print()

F={}
for vehicle_types in Vehicles_Specifications_VehicleType_arr:
    F[vehicle_types]=Vehicles_Specifications_WeightCapacity_dict[vehicle_types]
for pickup_cargos in PU:
    F[pickup_cargos]=Cargo_PickUp_UnitWeight_dict[pickup_cargos]
for delivery_cargos in DY:
    F[delivery_cargos]=Cargo_Delivery_UnitWeight_dict[delivery_cargos]
#print(F)


OT=[]

#print(Vehicles_Specifications_OpenTour_dict)
for vehicle_types in Vehicles_Specifications_VehicleType_arr:
    if Vehicles_Specifications_OpenTour_dict[vehicle_types] == "Yes":
        #print(vehicle_types," : ",Vehicles_Specifications_OpenTour_dict[vehicle_types])
        OT.append(vehicle_types)
    elif Vehicles_Specifications_OpenTour_dict[vehicle_types] != "No":
        raise Exception('Only "Yes" and "No" is allowed when user inputs the open-tour requirements. \n Check all entries in \n File:"0 Vehicles" \t Sheet:"Specifications" \t Column:"Need for return to the starting Depot"')


print(OT)


# # GRIP starts here


start_time=time.time()


# # Incorporate these PreProcessings:
# 
# # 1) Reject any null nodes (Nodes which have nothing to pickup or deliver)
# The above is already done while creating the set unsatisfied_nodes
# 
# # 2) Check initial available cargo feasibility of the problem (like sum of all Warehouse availabilities are greater than the demand at all Nodes, and sum of all pickup quantities are lesser than the cumulative Relief Centres capabilities)


N



NM



Un_Satisfied_Simultaneous_Nodes={}
for simultaneous_node in NM:
    Un_Satisfied_Simultaneous_Nodes[simultaneous_node]={}
    for pickup_cargo in PU:
        Un_Satisfied_Simultaneous_Nodes[simultaneous_node][pickup_cargo]=LocationPickupDelivery_SimultaneousNodes_PickUpCargoAwaiting_dict[pickup_cargo][simultaneous_node]
    for delivery_cargo in DY:
        Un_Satisfied_Simultaneous_Nodes[simultaneous_node][delivery_cargo]=LocationPickupDelivery_SimultaneousNodes_DeliveryCargoRequired_dict[delivery_cargo][simultaneous_node]
#print(Un_Satisfied_Simultaneous_Nodes)


Un_Satisfied_Split_Nodes={}
for split_node in NP:
    Un_Satisfied_Split_Nodes[split_node]={}
    for pickup_cargo in PU:
        Un_Satisfied_Split_Nodes[split_node][pickup_cargo]=LocationPickupDelivery_SplitNodes_PickUpCargoAwaiting_dict[pickup_cargo][split_node]
    for delivery_cargo in DY:
        Un_Satisfied_Split_Nodes[split_node][delivery_cargo]=LocationPickupDelivery_SplitNodes_DeliveryCargoRequired_dict[delivery_cargo][split_node]
#print(Un_Satisfied_Split_Nodes)


dynamic_Warehouse_status={}
for warehouse in W:
    dynamic_Warehouse_status[warehouse]={}
    for delivery_cargo in DY:
        dynamic_Warehouse_status[warehouse][delivery_cargo]=LocationPickupDelivery_Warehouses_DeliveryCargoAvailable_dict[delivery_cargo][warehouse]
#print(dynamic_Warehouse_status)



dynamic_ReliefCentre_status={}
for relief_centre in RC:
    dynamic_ReliefCentre_status[relief_centre]={}
    for pickup_cargo in PU:
        dynamic_ReliefCentre_status[relief_centre][pickup_cargo]=LocationPickupDelivery_ReliefCentres_PickUpCargoSpaceAvailable_dict[pickup_cargo][relief_centre]
#print(dynamic_ReliefCentre_status)


# # Finding only those Modes of Transport that can form routes and have atleast one Vehicle Type plying


_2_MultiModality_for_Distance_Matrix=pd.read_excel("2 MultiModality for Distance Matrix.xlsx",sheet_name=None, index_col=0)
#_2_MultiModality_for_Distance_Matrix=pd.read_excel("/content/drive/MyDrive/Original_dataset_for_objective2 - Copy/Basic_Instances/Instance2/2 MultiModality for Distance Matrix.xlsx",sheet_name=None, index_col=0)
#print(_2_MultiModality_for_Distance_Matrix)

Allowable_Modes=[] # To store which Modes are significant
for sheet in _2_MultiModality_for_Distance_Matrix:

    mode_key_value_setofVehicleTypes={}

    for modal_column_heading in _2_MultiModality_for_Distance_Matrix[sheet]:

        mode_key_value_setofVehicleTypes[modal_column_heading]=[]
        does_any_vehicle_work_in_this_mode=0 #Binary decider indicating if any vehicle type works in this mode

        for vehicle_type in Vehicles_Specifications_VehicleType_arr:
            if _2_MultiModality_for_Distance_Matrix[sheet][modal_column_heading][vehicle_type]:

                does_any_vehicle_work_in_this_mode=1
                mode_key_value_setofVehicleTypes[modal_column_heading].append(vehicle_type)
                #break


        min_route_formation_possible_in_this_mode= 0
        modal_compatibility_with_atleast_1_W= 0
        modal_compatibility_with_atleast_1_TP= 0
        modal_compatibility_with_atleast_1_N= 0
        modal_compatibility_with_atleast_1_RC= 0
        for vertex in V_0: # Assuming that checks for the VD are not required since they would naturally be as per their Vehicle Types
            if _2_MultiModality_for_Distance_Matrix[sheet][modal_column_heading][vertex]:
                if vertex in W:
                    modal_compatibility_with_atleast_1_W=1
                elif vertex in TP:
                    modal_compatibility_with_atleast_1_TP=1
                elif vertex in N:
                    modal_compatibility_with_atleast_1_N=1
                elif vertex in RC:
                    modal_compatibility_with_atleast_1_RC=1

                if modal_compatibility_with_atleast_1_W+modal_compatibility_with_atleast_1_TP+modal_compatibility_with_atleast_1_N+modal_compatibility_with_atleast_1_RC - (modal_compatibility_with_atleast_1_W*modal_compatibility_with_atleast_1_RC) >=2:
                    min_route_formation_possible_in_this_mode = 1
                    break

        if does_any_vehicle_work_in_this_mode*min_route_formation_possible_in_this_mode:
            Allowable_Modes.append(modal_column_heading)






print(Allowable_Modes)
# print(column_heading)
# print(_2_MultiModality_for_Distance_Matrix[sheet][column_heading])
# print(_2_MultiModality_for_Distance_Matrix[sheet][column_heading]["RC3"])



import random

print(mode_key_value_setofVehicleTypes)

Vehicles_Specifications_VolumeCapacity_dict[random.choice(mode_key_value_setofVehicleTypes["Road"])]



# import random
# from itertools import cycle
# # lst = ['a', 'b', 'c']
# # pool = cycle(lst)
# # for item in pool:
# #     print(item)

# # Starting Check if there are any Unsatisfied Nodes
# if Un_Satisfied_Simultaneous_Nodes or Un_Satisfied_Split_Nodes or Un_Satisfied_Transhipment_Ports:


#     Chosen_Node = random.choice(list(Un_Satisfied_Simultaneous_Nodes.keys()) + list(Un_Satisfied_Split_Nodes.keys()) + list(Un_Satisfied_Transhipment_Ports.keys()))

#     modes_available_to_Chosen_Node = _2_MultiModality_for_Distance_Matrix["MultiModal Interactivity"].loc[Chosen_Node]
#     significant_mode_iterator=[]

#     for mode,is_available in modes_available_to_Chosen_Node.iteritems():
#         if is_available and mode in Allowable_Modes:
#             significant_mode_iterator.append(mode)

#     for Chosen_Mode in cycle(significant_mode_iterator):

#         Chosen_VehicleType = random.choice(mode_key_value_setofVehicleTypes[Chosen_Mode])

#         if Chosen_Node in Un_Satisfied_Simultaneous_Nodes:

#             Required_Minimum_Vehicle_Capacity_WEIGHT=0
#             Required_Minimum_Vehicle_Capacity_VOLUME=0
#             Random_Vehicle_Type_Choices_within_this_mode =


#             while Required_Minimum_Vehicle_Capacity_VOLUME < E[Chosen_VehicleType] and Required_Minimum_Vehicle_Capacity_WEIGHT < F[Chosen_VehicleType]:



#                 for DYPU in Un_Satisfied_Simultaneous_Nodes[Chosen_Node]:
#                     Required_Minimum_Vehicle_Capacity_WEIGHT += F[DYPU]*Un_Satisfied_Simultaneous_Nodes[Chosen_Node][DYPU]
#                     Required_Minimum_Vehicle_Capacity_VOLUME += E[DYPU]*Un_Satisfied_Simultaneous_Nodes[Chosen_Node][DYPU]

#                 if Required_Minimum_Vehicle_Capacity_VOLUME > E[Chosen_VehicleType] or Required_Minimum_Vehicle_Capacity_WEIGHT > F[Chosen_VehicleType]:
#                     Chosen_VehicleType = random.choice(mode_key_value_setofVehicleTypes[Chosen_Mode]) # Choosing a New Vehicle Type, if the previous Vheicle Type was unable to take the full load of this Simultaneous Node
#                     Required_Minimum_Vehicle_Capacity_WEIGHT=0
#                     Required_Minimum_Vehicle_Capacity_VOLUME=0

#                 else:








# #         Max_Possible_vehicle_loading_Weight = Vehicles_Specifications_WeightCapacity_dict[Chosen_VehicleType]
# #         E = Volume
# #         F = Weight
# #         Max_Possible_vehicle_loading_Volume = Vehicles_Specifications_VolumeCapacity_dict[Chosen_VehicleType]



#         #for Chosen_VehicleType in cycle(mode_key_value_setofVehicleTypes):






#     if Chosen_Node in Un_Satisfied_Simultaneous_Nodes:

#         # Filter the DataFrame to get the columns where the value in 'desired_row_heading' is 1
#         #columns_with_ones = df.columns[df[desired_row_heading] == 1].tolist()
#         #columns_with_ones = _2_MultiModality_for_Distance_Matrix.columns[_2_MultiModality_for_Distance_Matrix[Chosen_Node] == 1].tolist()
#         #row = _2_MultiModality_for_Distance_Matrix["MultiModal Interactivity"].loc[Chosen_Node]

#         pass

#     elif Chosen_Node in Un_Satisfied_Split_Nodes:
#         pass







# else:
#     print("All Nodes have been incorporated as routes")
#     print("Now Optimality Iterations (Perturbations) and \n Assignment to exact Vehicles (completing routes to Vehicle Depots) should be done")


# g={}
# h={}

# if g or h:
#     print("jui")



# for vehicle_type in Vehicles_Specifications_VehicleType_arr:
#     if _2_MultiModality_for_Distance_Matrix["MultiModal Interactivity"]["Air"][vehicle_type]:
#         does_any_vehicle_work_in_this_mode=1





print(_2_MultiModality_for_Distance_Matrix["MultiModal Interactivity"].loc["NM1"])

# for h,i in _2_MultiModality_for_Distance_Matrix["MultiModal Interactivity"].loc["NM1"].iteritems():
#     print(h," : ",i)


for vehicle_types in Vehicles_Specifications_VehicleType_arr:
    print(vehicle_types)



Un_Satisfied_Simultaneous_Nodes


list(Un_Satisfied_Simultaneous_Nodes.keys()) + list(Un_Satisfied_Split_Nodes.keys())


# # Putting all the unsatisfied nodes in a single set (only thse node which have non-zero loadcode)


# Initialize a dynamic set to store unsatisfied nodes
unsatisfied_nodes = set()

# Add unsatisfied simultaneous nodes to the set
for simultaneous_node in NM:
    for pickup_cargo in PU:
        if Un_Satisfied_Simultaneous_Nodes[simultaneous_node][pickup_cargo] > 0:
            unsatisfied_nodes.add(simultaneous_node)
    for delivery_cargo in DY:
        if Un_Satisfied_Simultaneous_Nodes[simultaneous_node][delivery_cargo] > 0:
            unsatisfied_nodes.add(simultaneous_node)

# Add unsatisfied split nodes to the set
for split_node in NP:
    for pickup_cargo in PU:
        if Un_Satisfied_Split_Nodes[split_node][pickup_cargo] > 0:
            unsatisfied_nodes.add(split_node)
    for delivery_cargo in DY:
        if Un_Satisfied_Split_Nodes[split_node][delivery_cargo] > 0:
            unsatisfied_nodes.add(split_node)

# Print or use the unsatisfied_nodes set as needed
print("Unsatisfied Nodes:", unsatisfied_nodes)

dynamic_set=[]
for element in unsatisfied_nodes:
    dynamic_set.append(element)
print(dynamic_set)



def calculate_distance(node1, node2,vt):
    if node1 == node2:
        print("Fatal Error")
        #print(Fatal_Error)
    return C[node1,node2,vt]


# Initialize a dictionary to store routes for each vehicle type
routes_by_vehicle_type = {vehicle_type: [] for vehicle_type in Vehicles_Specifications_VehicleType_arr}

print(routes_by_vehicle_type)



def can_access_vertex_or_vehicleType_multimodal(transportation_mode, vertex_or_vehicleType):
#     if transportation_mode == "Rail":
#         return Multimode_Rail_dict[node] == 1
#     elif transportation_mode == "Road":
#         return Multimode_Road_dict[node] == 1
#     else:
#         return False

    return _2_MultiModality_for_Distance_Matrix["MultiModal Interactivity"][transportation_mode][vertex_or_vehicleType] == 1
    
    
#can_access_vertex_or_vehicleType_multimodal("Air", "NM10")


# # Next, we can alter the following function with not only the nearest warehouse selection but also weighted time selections, like:
# 1) Time weighted with the ratio of the no. of types of cargos (partially or fully) satisfiable at the Warehouse wrt. the total non-zero cargo required at the Node
# 
# 2) Time weighted with ratio of the sum of amounts of all types of cargos which can be satisfied at the Warehouse wrt. the total requirement at the Node (by units, mass or volume)
# 
# 3) Time weighted with the ratio of the increase in the vehicle load (V/M) by the Warehouse wrt. the maximum increase in vehicle load (volume/mass) by all Warehouses


def find_nearest_warehouse(current_node, Q, mode, selected_vehicle_type, node_required_delivery):
    warehouses_with_enough_quantities = [
        warehouse
        for warehouse in W
        if any((node_required_delivery[cargo]>0 #Whether this cargo is required at Node
               and Q[cargo, warehouse]>0 #Whether the required cargo is available at the Warehouse
               and VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[cargo][selected_vehicle_type]!=-1 #Vehicle is not incompatible to carry the selected cargo
               and can_access_vertex_or_vehicleType_multimodal(mode,warehouse)) #The Warehouse is accessible to the Mode of transport
               for cargo in DY)
    ]
    
    if warehouses_with_enough_quantities:
        # Find the nearest warehouse among those with enough quantities
        nearest_warehouse = min(
              warehouses_with_enough_quantities,
              key=lambda warehouse: calculate_distance(current_node, warehouse, selected_vehicle_type),
          ) # Alteration here
        return nearest_warehouse
       
    else:
        return None
        #This means that there is no Warehouse which can be visited by this Vehicle Type
        #and if this is the case for all Vehicle Types,
        #then a Transhipment Port is a must use



def find_nearest_relief_centre(current_node, Q, mode, selected_vehicle_type, node_required_pickup):
    relief_centres_with_enough_quantities = [
        rc
        for rc in RC
        if any((node_required_pickup[cargo]>0
               and Q[cargo, rc]>0
               and VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[cargo][selected_vehicle_type]!=-1
               and can_access_vertex_or_vehicleType_multimodal(mode,rc))
               for cargo in PU)
    ]
        
    if relief_centres_with_enough_quantities:
        # Find the nearest warehouse among those with enough quantities
        nearest_relief_centre = min(
              relief_centres_with_enough_quantities,
              key=lambda rc: calculate_distance(current_node, rc, selected_vehicle_type),
          )
        return nearest_relief_centre
    
    else:
        return None #This means that there is no Relief Centre which can be visited by this Vehicle Type and if this is the case for all Vehicle Types, then a Transhipment Port is a must use



def nearest_TranSShipmentPort(current_vertex, Q, mode, selected_vehicle_type, requirements_at_the_node):
    transhipment_port_compatible = [
        tp
        for tp in TP
        if any((requirements_at_the_node[cargo]>0 #Whether this cargo is required at Node
           and VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[cargo][selected_vehicle_type]!=-1 #Vehicle is not incompatible to carry the selected cargo
           and can_access_vertex_or_vehicleType_multimodal(mode,tp)) #The Warehouse is accessible to the Mode of transport
           for cargo in DY+PU)]
    
    
    
    pass



DSED=[1,2,3]
DSED2 = [8,9]

print(DSED+DSED2)



def find_nearest_warehouse_for_split_node(node, Q, mode, vehicle_type):
    node_required_delivery = {cargo: Q[cargo, node] for cargo in DY}
    warehouses_with_enough_quantities = [
        warehouse
        for warehouse in W
        if any((node_required_delivery[cargo]>0
               and Q[cargo, warehouse]>0
               and VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[cargo][vehicle_type]!=-1
               and can_access_vertex_or_vehicleType_multimodal(mode,warehouse))
               for cargo in DY)
    ]

    if warehouses_with_enough_quantities:
        # Find the nearest warehouse among those with enough quantities
        nearest_warehouse = min(
            warehouses_with_enough_quantities,
            key=lambda warehouse: calculate_distance(node, warehouse, vehicle_type),
        )
        return nearest_warehouse
         
    else:
        return False
        #This means that there is no Warehouse which can be visited by this Vehicle Type
        #and if this is the case for all Vehicle Types, then a Transhipment Port is a must use


def find_nearest_relief_centre_for_split_node(node, Q, mode, vehicle_type):
    node_required_pickup = {cargo: Q[cargo, node] for cargo in PU}

    # Create a list of rcs with enough free space for at least one cargo
    relief_centres_with_enough_quantities = [
        rc
        for rc in RC
        if any((node_required_pickup[cargo]>0
               and Q[cargo, rc]>0
               and VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[cargo][vehicle_type]!=-1
               and can_access_vertex_or_vehicleType_multimodal(mode,rc))
               for cargo in PU)
    ]
    
    if relief_centres_with_enough_quantities:
        # Find the nearest relief center among those with enough free-space
        nearest_relief_centre = min(
            relief_centres_with_enough_quantities,
            key=lambda rc: calculate_distance(node, rc, vehicle_type),
        )
        return nearest_relief_centre
      
    else:
        return False
        #This means that there is no Relief Centre which can be visited by this Vehicle Type
        #and if this is the case for all Vehicle Types, then a Transhipment Port is a must use



def find_satisfied_quantity_for_split_node(node, nearest_warehouse, nearest_relief_centre, Q, vehicle_type):
    satisfied_quantity={}
    print(nearest_warehouse)
    print(nearest_relief_centre)
    
    rem_weight_capacity_vehicle=F[vehicle_type]
    rem_volume_capacity_vehicle=E[vehicle_type]
    if nearest_warehouse!=False:
        
        random.shuffle(DY)
        for cargo in DY:
            print(cargo)
            print(Q[cargo,node])
            print(Q[cargo,nearest_warehouse])
            x=min(Q[cargo,node],Q[cargo,nearest_warehouse])
            if VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[cargo][vehicle_type]==-1: # No cargo amount is carried if the vehicle is incompatible to carry it
                satisfied_quantity[cargo]=0
                continue
                
            if E[cargo]==0 or F[cargo]==0:
                # The "or" may also be replaced with an "and" above, but the "or" is retained as it will also enable users to see massive allocation incase any of the capacities are assigned 0 mistakenly
                # If the cargo does not have any capacity attached, then the entire cargo is considered to be satisfied
                satisfied_quantity[cargo]=x
                rem_weight_capacity_vehicle-=x*F[cargo]
                rem_volume_capacity_vehicle-=x*E[cargo]
                continue
            x=min(x,rem_volume_capacity_vehicle//E[cargo]) #WRONG corrected here
            x=min(x,rem_weight_capacity_vehicle//F[cargo])
            rem_weight_capacity_vehicle-=x*F[cargo]
            rem_volume_capacity_vehicle-=x*E[cargo]
            satisfied_quantity[cargo]=x
    else:
        for cargo in DY:
            satisfied_quantity[cargo]=0
            
    rem_weight_capacity_vehicle=F[vehicle_type]
    rem_volume_capacity_vehicle=E[vehicle_type]
    if nearest_relief_centre!=False:
        
        random.shuffle(PU)
        for cargo in PU:
            print(cargo)
            print(Q[cargo,node])
            print(Q[cargo,nearest_relief_centre])
            x=min(Q[cargo,node],Q[cargo,nearest_relief_centre])  # Keep track of remaining capacity of vehicle
            if VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[cargo][vehicle_type]==-1:
                satisfied_quantity[cargo]=0
                continue
            if E[cargo]==0 or F[cargo]==0:
                # The "or" may also be replaced with an "and" above, but the "or" is retained as it will also enable users to see massive allocation incase any of the capacities are assigned 0 mistakenly
                # If the cargo does not have any capacity attached, then the entire cargo is considered to be satisfied
                satisfied_quantity[cargo]=x
                rem_weight_capacity_vehicle-=x*F[cargo]
                rem_volume_capacity_vehicle-=x*E[cargo]
                continue
            x=min(x,rem_volume_capacity_vehicle//E[cargo])
            x=min(x,rem_weight_capacity_vehicle//F[cargo]) #WRONG corrected here as well
            rem_weight_capacity_vehicle-=x*F[cargo]
            rem_volume_capacity_vehicle-=x*E[cargo]
            satisfied_quantity[cargo]=x
    else:
        for cargo in PU:
            satisfied_quantity[cargo]=0
    
    return satisfied_quantity


# # Comment:
# 
# It may be felt that instead of required_delivery below, shouldn't they be satisfied_quantity? Since the below functions are specifically for Simultaneous Nodes, the satisfied quantity must be the same as the required delivery, therefore it's not a problem but is actually crrect.
# 
# # For Simultaneous Nodes only


# Function to update capacities after satisfying a Simultaneous node's needs
# (single Vehicle visit at the Simultaneous Node requires the vehicle capacity to be enough to fulfill everything at one go)
def update_capacities_warehouses(warehouse_id, required_delivery, Q):
    # random.shuffle(DY) connot use shuffle here as its complicating many things; better to use outside of the if node_id in NM condition
    # No need to shuffle actually as vehicle must has enough space
    for cargo in DY:
        x=min(Q[cargo,warehouse_id],required_delivery[cargo])
        Q[cargo,warehouse_id]-=x
        required_delivery[cargo]-=x


# For Simultaneous Nodes only
def update_capacities_relief_centres(relief_id, required_pickup, Q):
    # random.shuffle(PU) connot use shuffle here as its complicating many things; better to use outside of the if node_id in NM condition
    # No need to shuffle actually as vehicle must has enough space
    for cargo in PU:
        x=min(Q[cargo,relief_id],required_pickup[cargo])
        Q[cargo,relief_id]-=x
        required_pickup[cargo]-=x


# For Simultaneous Nodes only
def barcode_warehouse(warehouse_id, required_delivery, Q):
    barcode={}
    for cargo in DY:
        barcode[cargo]=min(Q[cargo,warehouse_id],required_delivery[cargo])
    for cargo in PU:
        barcode[cargo]=0
    return barcode


# For Simultaneous Nodes only
def barcode_relief_centre(relief_id, required_pickup, Q):
    barcode={}
    for cargo in DY:
        barcode[cargo]=0
    for cargo in PU:
        barcode[cargo]=-1*min(Q[cargo,relief_id],required_pickup[cargo])
    return barcode



# For Simultaneous Nodes only as they would require the entire cargo requirements to be shipped into and out together
def can_carry_load(vehicle_type, Simultaneous_node_id, Q):
        
    if any((VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[cargo][vehicle_type]==-1
            and Q[cargo, Simultaneous_node_id]>0) for cargo in DY):
        return False
        
    if any((VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[cargo][vehicle_type]==-1
            and Q[cargo, Simultaneous_node_id]>0) for cargo in PU):
        return False
        
    #node_required_quantities = {cargo: Q[cargo,node_id] for cargo in DY}
    total_weight_DY=0
    total_volume_DY=0
    for cargo in DY:
        total_weight_DY+=Q[cargo, Simultaneous_node_id]*F[cargo]
        total_volume_DY+=Q[cargo, Simultaneous_node_id]*E[cargo]
        
    if total_weight_DY>F[vehicle_type] or total_volume_DY>E[vehicle_type]:
        return False
    
    total_weight_PU=0
    total_volume_PU=0
    for cargo in PU:
        total_weight_PU+=Q[cargo, Simultaneous_node_id]*F[cargo]
        total_volume_PU+=Q[cargo, Simultaneous_node_id]*E[cargo]
    if total_weight_PU>F[vehicle_type] or total_volume_PU>E[vehicle_type]:
        return False
        
    return True



# Function to satisfy nodes until the dynamic set is empty
def satisfy_nodes(dynamic_set, Q):
    no_solution=False #Unused as yet
    need_to_be_satisfied_by_transhipment=[]
    while dynamic_set:
        route_found=False
        node_id = random.choice(dynamic_set)
        print("Randomly chosen Node to be satisfied: ", node_id)
        
        # Select modes of transport that can access the random node
        transport_modes = [mode for mode in Allowable_Modes if can_access_vertex_or_vehicleType_multimodal(mode, node_id)]

        tabu_list_of_chosen_modes=set()        
        while transport_modes:
            mode=random.choice(transport_modes)
            tabu_list_of_chosen_modes.add(mode)
            transport_modes.remove(mode) # May be added again if the chosen Node is of NP type
            
            print("Randomly chosen Mode: ", mode)
            
            # Find available vehicle types for the selected mode of transport
            available_vehicle_types = []
            for vh in Vehicles_Specifications_VehicleType_arr:
                if can_access_vertex_or_vehicleType_multimodal(mode,vh):
                    available_vehicle_types.append(vh)
            
#             if mode=='Rail':
#                 for vh in Vehicles_Specifications_VehicleType_arr:
#                     if Multimode_Rail_dict[vh]==1:
#                         available_vehicle_types.append(vh)

#             if mode=='Road':
#                 for vh in Vehicles_Specifications_VehicleType_arr:
#                     if Multimode_Road_dict[vh]==1:
#                         available_vehicle_types.append(vh)
            
            
            tabu_list_of_chosen_vehicles=set()
            while available_vehicle_types:
                vehicle_type=random.choice(available_vehicle_types)
                tabu_list_of_chosen_vehicles.add(vehicle_type)
                available_vehicle_types.remove(vehicle_type)
                print("Randomly chosen Vehicle Type: ", vehicle_type)
                
                
                # For simultaneous nodes, find the nearest warehouse and relief centre
                if node_id in NM:
                    
                    if can_carry_load(vehicle_type, node_id, Q): # Checks if the Vehicle capacities are enough for satisfying
                        #the needs of the Simultaneous Node's requirements
                        
                        required_delivery = {cargo: Q[cargo,node_id] for cargo in DY}
                        required_pickup = {cargo: Q[cargo,node_id] for cargo in PU}
                        print("Required Delivery: ", required_delivery)
                        print("Required PickUp: ", required_pickup)
                        curr_node=node_id
                        route={}
                        barcode_at_node={}
                        for cargo in DY:
                            barcode_at_node[cargo]=-1*required_delivery[cargo]
                        for cargo in PU:
                            barcode_at_node[cargo]=required_pickup[cargo]
                        
                        Q_backup=Q.copy()
                        required_delivery_backup=required_delivery.copy()
                        required_pickup_backup=required_pickup.copy()
                        
                        warehouse_id = "NotNone"
                        while any(required_delivery[cargo]>0 for cargo in DY):
                        
                            warehouse_id=find_nearest_warehouse(curr_node,Q,mode,vehicle_type,required_delivery)
                            
                            if warehouse_id == None: # Indicates Vehicle Type change is necessary due to Vehicle~Cargo Compatibility
                                break
                                
                            barcode=barcode_warehouse(warehouse_id,required_delivery,Q)
                            route[warehouse_id]=barcode
                            update_capacities_warehouses(warehouse_id,required_delivery,Q)
                            curr_node=warehouse_id
                            
                        if warehouse_id == None:  # Indicates Vehicle Type change is necessary due to Vehicle~Cargo Compatibility
                            Q=Q_backup.copy()
                            required_delivery=required_delivery_backup.copy()
                            #required_pickup=required_pickup_backup.copy()
                            continue
                        
                        route[node_id]=barcode_at_node
                        curr_node=node_id
                        
                        relief_id = "NotNone"
                        while any(required_pickup[cargo]>0 for cargo in PU):
                            relief_id=find_nearest_relief_centre(curr_node,Q,mode,vehicle_type,required_pickup)
                            
                            if relief_id == None: # Indicates Vehicle Type change is necessary due to Vehicle~Cargo Compatibility
                                break
                                
                            barcode=barcode_relief_centre(relief_id,required_pickup,Q)
                            route[relief_id]=barcode
                            update_capacities_relief_centres(relief_id,required_pickup,Q)
                            curr_node=relief_id
                            
                        if relief_id == None: # Indicates Vehicle Type change is necessary due to Vehicle~Cargo Compatibility
                            Q=Q_backup.copy()
                            required_delivery=required_delivery_backup.copy()
                            required_pickup=required_pickup_backup.copy()
                            continue
                                
                        print("Generated route: ", route)
                        print("Q: ", Q)
                        dynamic_set.remove(node_id)
                        route_found=True
                        routes_by_vehicle_type[vehicle_type].append(route)
                        break  # Move to the next node in the dynamic set
                        
                # For split nodes, satisfy as much as possible with revised requirements
                elif node_id in NP:
                    if any((Q[cargo,node_id]>0 and VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[cargo][vehicle_type]!=-1) for cargo in DY) or any((Q[cargo,node_id]>0 and VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[cargo][vehicle_type]!=-1) for cargo in PU):
                        a=False
                        for cargo in DY:
                            if Q[cargo,node_id]>0 and VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[cargo][vehicle_type]!=-1:
                                a=True
                        if a==True:
                            warehouse_id=find_nearest_warehouse_for_split_node(node_id,Q,mode,vehicle_type)
                        else:
                            warehouse_id=False
                            
                        a=False
                        for cargo in PU:
                            if Q[cargo,node_id]>0 and VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[cargo][vehicle_type]!=-1:
                                a=True
                        if a==True:
                            relief_id=find_nearest_relief_centre_for_split_node(node_id,Q,mode,vehicle_type)
                        else :
                            relief_id=False
                        
                        
                        if not (relief_id or warehouse_id):#If both ids are False, we continue the loop for another Vehicle Type
                            # (since either the Vehicle Type was incompatible or the mode needs to be changed)
                            #available_vehicle_types.remove(vehicle_type)
                            continue
                        
                        
                        
                        satisfied_quantity=find_satisfied_quantity_for_split_node(node_id,warehouse_id,relief_id,Q,vehicle_type)
                        print("Warehouse: ", warehouse_id)
                        print("Relief Centre: ", relief_id)
                        print("Satisfied Quantity: ", satisfied_quantity)
                        #print(warehouse_id,relief_id)

                        # Update revised requirements with the remaining quantity
                        rem=False
                        if warehouse_id:
                            for cargo in DY:
                                Q[cargo,node_id]-=satisfied_quantity[cargo]
                                Q[cargo,warehouse_id]-=satisfied_quantity[cargo]
                                if Q[cargo,node_id]>0:
                                    rem=True
                        if relief_id:
                            for cargo in PU:
                                Q[cargo,node_id]-=satisfied_quantity[cargo]
                                Q[cargo,relief_id]-=satisfied_quantity[cargo]
                                if Q[cargo,node_id]>0:
                                    rem=True
                        
                        # If all needs are satisfied, remove the node from the dynamic set
                        if rem == False:
                            dynamic_set.remove(node_id)

                        route_found=True
                        barcode_at_node={}
                        barcode_at_warehouse={}
                        barcode_at_relief_centre={}
                        for cargo in DY:
                            barcode_at_node[cargo]=-1*satisfied_quantity[cargo]
                            barcode_at_warehouse[cargo]=satisfied_quantity[cargo]
                            barcode_at_relief_centre[cargo]=0
                        for cargo in PU:
                            barcode_at_node[cargo]=satisfied_quantity[cargo]
                            barcode_at_warehouse[cargo]=0
                            barcode_at_relief_centre[cargo]=-1*satisfied_quantity[cargo]
                        if warehouse_id==False:
                            route = {node_id:barcode_at_node,relief_id:barcode_at_relief_centre}
                        elif relief_id==False:
                            route = {warehouse_id:barcode_at_warehouse,node_id:barcode_at_node}
                        else:
                            route = {warehouse_id:barcode_at_warehouse,node_id:barcode_at_node,relief_id:barcode_at_relief_centre}
                        print("Route: ",route)
                        print("Q: ",Q)
                        routes_by_vehicle_type[vehicle_type].append(route)
                        break
                    
                    #else:
                    #This else activates when the Vehicle is incompatible to carry any load of the chosen Node,
                    #or if there is no load to be carried
                        #(The case of all zero cargo is neglected as such a node should be removed during pre-processing
                        #and is removed as well in this function)
                        #available_vehicle_types.remove(vehicle_type)
                else:
                    print("Node has to be either in NP or NM")
                    
            if route_found==True:
                break
            elif no_solution==True:
                break
            #elif available_vehicle_types:
            
            
            #else
            if transport_modes:
                continue
            else: # if the computer is here, then this would mean that all the modes have been considered
                # and therefore all vehicle types within them have also been considered
                #Create a Transhipment to satisfy this Node
                need_to_be_satisfied_by_transhipment.append(node_id)
                dynamic_set.remove(node_id)
                
                
            
        
            
        print("Dynamic Array: ", dynamic_set)
        print("Nodes that must be Satisfied by Transhipment: ",need_to_be_satisfied_by_transhipment)


# # Transhipment Port inclusion in logic


def satisfy_nodes_considering_transhipment(need_to_be_satisfied_by_transhipment):
    
    # We assign one Transhipment Port to reduce loading and unloading
    
    # For the Simultaneous Ports (NM) we may rerun the assignment (in reverse) allowing atleast the necessary assignment into transhipment ports and more with some probability
    
    # For Split Node (NP) we assign the entire amounts to the nearest Transhipment Port with weighted probability to also choose different
    
    node_arr_to_only_satisfy_by_transhipment = need_to_be_satisfied_by_transhipment.copy()
    
    while node_arr_to_only_satisfy_by_transhipment:
        
        route_found=False
        node_id = random.choice(node_arr_to_only_satisfy_by_transhipment)
        print("Randomly chosen Node to be satisfied by Transhipment: ", node_id)
        node_arr_to_only_satisfy_by_transhipment.remove(node_id)
        
        




            
            
        #else:
        #    print("Node has to be either in NP or NM")
                
        
        
        


# # The above function is developed such that Transhipment occurs only when it is absolutely necessary. This may prove sub-optimal in simulated instances, but we feel that in real world cases, especially during disasters, multiple unloading+loading may not be fruitful (and could aswell be time consuming)
# 
# # Route Creation: Greedy approach as we try to satisfy the Nodes with their nearest resource/relief points
# 
# # The routes are developed specific to Vehicle Types


satisfy_nodes(dynamic_set, Q)



print(dynamic_set)


print(routes_by_vehicle_type)


def times(route,vehicle_type,curr_route): #route refers to the new route which is to be inserted after the curr_route
    curr_node=curr_route[1] # This is the last node of the current route, i.e. the current vehicle location
    curr_time=curr_route[0]
    for ele in route:
        curr_time+=C[curr_node,ele,vehicle_type]
        curr_node=ele
        curr_dict=route[ele]
        for cargo in curr_dict:
            if cargo in DY:
                curr_time+=VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[cargo][vehicle_type]*abs(curr_dict[cargo])
            else:
                curr_time+=VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[cargo][vehicle_type]*abs(curr_dict[cargo])

    return curr_time,curr_node


dynamic_time={}

vehicle_depots_in_vehicle_type={}

for Vehicle_Type in Vehicles_Specifications_VehicleType_arr:
    vehicle_depots_in_vehicle_type[Vehicle_Type]=[]

print("Vehicle Depots for Vehicle Types: ", vehicle_depots_in_vehicle_type)

for vehicle_depot,vehicle_type in vN:
    vehicle_depots_in_vehicle_type[vehicle_type].append(vehicle_depot)
    for vehicle_no in range(1,vN[vehicle_depot,vehicle_type]+1):
        dynamic_time[vehicle_type,vehicle_depot,vehicle_no]=[0,vehicle_depot]
    
print("Vehicle Depots for Vehicle Types: ", vehicle_depots_in_vehicle_type)
print("Dynamic Time: ", dynamic_time)


list_of_routes={}

for vehicle_type in Vehicles_Specifications_VehicleType_arr:
    for vehicle_depot in vehicle_depots_in_vehicle_type[vehicle_type]:
        for vehicle_no in range(1,vN[vehicle_depot,vehicle_type]+1):
            list_of_routes[vehicle_type,vehicle_depot,vehicle_no]=[]

print(list_of_routes)


# # Develop a perfectly bigger Vehicle Dictionary during Route Assignment/Allocation (with similar or broader compatibility of Cargo Carrying)


route_allocation={}
for vehicle_type in Vehicles_Specifications_VehicleType_arr:
    print("Vehicle Type: ", vehicle_type)
    for route in routes_by_vehicle_type[vehicle_type]:
        mini=1e9
        vd=False
        vh_no=False
        last_node=False
        for vehicle_depot in vehicle_depots_in_vehicle_type[vehicle_type]:
            for vehicle_no in range(1,vN[vehicle_depot,vehicle_type]+1):
                t,l_node=times(route,vehicle_type,dynamic_time[vehicle_type,vehicle_depot,vehicle_no])
                if t<mini:
                    mini=t
                    vd=vehicle_depot
                    vh_no=vehicle_no
                    last_node=l_node
        dynamic_time[vehicle_type,vd,vh_no]=[mini,last_node]
        list_of_routes[vehicle_type,vd,vh_no].append(route)
        print("Route: ", route)
        print("Dynamic time: ",dynamic_time[vehicle_type,vd,vh_no],"\n")
    
    if not vehicle_type in OT:
        for vehicle_depot in vehicle_depots_in_vehicle_type[vehicle_type]:
            for vehicle_no in range(1,vN[vehicle_depot,vehicle_type]+1):
                if dynamic_time[vehicle_type,vehicle_depot,vehicle_no][0]!=0:
                    dynamic_time[vehicle_type,vehicle_depot,vehicle_no][0]+=C[dynamic_time[vehicle_type,vehicle_depot,vehicle_no][1],vehicle_depot,vehicle_type]
                    dynamic_time[vehicle_type,vehicle_depot,vehicle_no][1]=vehicle_depot


# # In the above function, the final return to the Vehicle Depot for closed trips is included in the Dynamic Time but not in the actual route (easing the Perturbation process)


max_T_before_perturbation=0

for vehicle_type in Vehicles_Specifications_VehicleType_arr:
    for vehicle_depot in vehicle_depots_in_vehicle_type[vehicle_type]:
        for vehicle_no in range(1,vN[vehicle_depot,vehicle_type]+1):
            print("\n Vehicle ID: ", vehicle_type,",",vehicle_depot,",",vehicle_no)
            print("Total Route Time: ",dynamic_time[vehicle_type,vehicle_depot,vehicle_no][0],", Final Vertex: ",dynamic_time[vehicle_type,vehicle_depot,vehicle_no][1])
            
            if dynamic_time[vehicle_type,vehicle_depot,vehicle_no][0] > max_T_before_perturbation:
                max_T_before_perturbation = dynamic_time[vehicle_type,vehicle_depot,vehicle_no][0]
            
            # The below Line prints the detailed Barcodes representing what is happening at each Node during each visit
            #print(list_of_routes[vehicle_type,vehicle_depot,vehicle_no])
            
            string_route="\n Route Starts: "
            for individual_route in list_of_routes[vehicle_type,vehicle_depot,vehicle_no]:
                for i in individual_route.keys():
                    string_route+=i
                    string_route+="  -=>  "
                string_route+=" \n -=>  "
                    
            print(string_route[ : -15],"\n")
                  
print("Maximum Route Time before any Perturbation: ", max_T_before_perturbation," seconds/hours/units")


def final_time(route,vehicle_type,vehicle_depot):
    curr_node=vehicle_depot
    curr_time=0
    if vehicle_type in OT:
        for ele in route:
            for node in ele:
                    curr_time+=C[curr_node,node,vehicle_type]
                    curr_node=node
                    curr_dict=ele[node]
                    for cargo in curr_dict:
                        if cargo in DY:
                            curr_time+=VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[cargo][vehicle_type]*abs(curr_dict[cargo])
                        else:
                            curr_time+=VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[cargo][vehicle_type]*abs(curr_dict[cargo])
                    #print(curr_time)
    else :
        for ele in route:
            for node in ele:
                curr_time+=C[curr_node,node,vehicle_type]
                curr_node=node
                curr_dict=ele[node]
                for cargo in curr_dict:
                    if cargo in DY:
                        curr_time+=VehicleCompatibility_LoadingUnloadingTime_CargoDeliveryTime_dict[cargo][vehicle_type]*abs(curr_dict[cargo])
                    else:
                        curr_time+=VehicleCompatibility_LoadingUnloadingTime_CargoPickUpTime_dict[cargo][vehicle_type]*abs(curr_dict[cargo])
        curr_time+=C[curr_node,vehicle_depot,vehicle_type]
    return curr_time


import numpy as np

perturb_array={}

for vehicle_type in Vehicles_Specifications_VehicleType_arr:
    for vehicle_depot in vehicle_depots_in_vehicle_type[vehicle_type]:
        for vehicle_no in range(1,vN[vehicle_depot,vehicle_type]+1):
            cols=0
            perturb_array[vehicle_type,vehicle_depot,vehicle_no]=[]
            # print(len(list_of_routes['VT1','VD1',1]))
            for item in list_of_routes[vehicle_type,vehicle_depot,vehicle_no]: # Iten refers to a satisfied Route Segment in the Complete Route of the specific vehicle
                temp_array=[]
                for i in range(cols):
                    temp_array.append([False,False])
                i=0
                for key in item:
                    temp_array.append([key,item[key]])
                    i=i+1
                cols+=i
                perturb_array[vehicle_type,vehicle_depot,vehicle_no].append(temp_array)
            rows=len(perturb_array[vehicle_type,vehicle_depot,vehicle_no])
            # print(perturb_array['VT1','VD1',1])
            # print(rows)
            # print(cols)
            for row in range(rows):
                curr_size=len(perturb_array[vehicle_type,vehicle_depot,vehicle_no][row])
                # print(curr_size)
                while curr_size<cols:
                    perturb_array[vehicle_type,vehicle_depot,vehicle_no][row].append([False,False])
                    curr_size=curr_size+1
            #print(perturb_array[vehicle_type,vehicle_depot,vehicle_no])
            
#print(perturb_array['VT4','VD2',1])
# for vehicle_type in Vehicles_Specifications_VehicleType_arr:
#   for vehicle_depot in vehicle_depots_in_vehicle_type[vehicle_type]:
#     for vehicle_no in range(1,vN[vehicle_depot,vehicle_type]+1):
#       # print(len(perturb_array[vehicle_type,vehicle_depot,vehicle_no]))
#       # print("Ho sakta hai")
#       for i in range(len(perturb_array[vehicle_type,vehicle_depot,vehicle_no])):
#         # print(len(perturb_array[vehicle_type,vehicle_depot,vehicle_no][i]))
# print(perturb_array['VT2','VD4',1])
                    

def feasible(route,vehicle_type):
    current_quantity={}
    for cargo in DY:
        current_quantity[cargo]=0
    for cargo in PU:
        current_quantity[cargo]=0
    for i in range(len(route)):
        custom_dict=route[i]
        for key in custom_dict:

            # This key == False case should not arise
            if key==False:
                print("KEY IS FALSE")
                # break
                return None
                
            barcode_dictionary = custom_dict[key]
            for cargo in barcode_dictionary:
                current_quantity[cargo] += barcode_dictionary[cargo]
        curr_wt=0
        curr_vl=0
        for cargo in DY:
            curr_wt+=current_quantity[cargo]*F[cargo]
            curr_vl+=current_quantity[cargo]*E[cargo]
        for cargo in PU:
            curr_wt+=current_quantity[cargo]*F[cargo]
            curr_vl+=current_quantity[cargo]*E[cargo]
        if curr_wt>F[vehicle_type] or curr_vl>E[vehicle_type]:
            return False
    return True


print(C['NM2','NM1','VT1'])



max_T=0
perturbed_route_time_mapping={}


# # Allow Suboptimal Perturbations similar to Simulated Annealing?


for vehicle_type in Vehicles_Specifications_VehicleType_arr:
    for vehicle_depot in vehicle_depots_in_vehicle_type[vehicle_type]:
        for vehicle_no in range(1,vN[vehicle_depot,vehicle_type]+1):
            
            print("\n Perturbation for Vehicle UID: ",vehicle_type,",",vehicle_depot,",",vehicle_no,", with length of the Perturb Array: ",len(perturb_array[vehicle_type,vehicle_depot,vehicle_no])," (equal to the no. of rows)")
            
            if len(perturb_array[vehicle_type,vehicle_depot,vehicle_no])==0:
                perturbed_route_time_mapping[vehicle_type,vehicle_depot,vehicle_no]=0
                continue
                
            elif len(perturb_array[vehicle_type,vehicle_depot,vehicle_no])==1:
                route=[]
                for element in perturb_array[vehicle_type,vehicle_depot,vehicle_no][0]:
                    route.append({element[0]:element[1]})
                temp=final_time(route,vehicle_type,vehicle_depot)
                max_T=max(max_T,temp)
                perturbed_route_time_mapping[vehicle_type,vehicle_depot,vehicle_no]=temp
                
            else:
                initial_route=[]
                # print(list_of_routes[vehicle_type,vehicle_depot,vehicle_no])
                # print(perturb_array[vehicle_type,vehicle_depot,vehicle_no])
                for j in range(len(perturb_array[vehicle_type,vehicle_depot,vehicle_no][0])): # Columns
                    for i in range(len(perturb_array[vehicle_type,vehicle_depot,vehicle_no])): # Rows
                        if perturb_array[vehicle_type,vehicle_depot,vehicle_no][i][j][0]!=False:
                            initial_route.append({perturb_array[vehicle_type,vehicle_depot,vehicle_no][i][j][0]:perturb_array[vehicle_type,vehicle_depot,vehicle_no][i][j][1]})
                            break
                # print(initial_route)
                temp=final_time(initial_route,vehicle_type,vehicle_depot)
                x=0
                # print(temp)
                while x<1000:
                    random_number_x = random.randint(0,len(perturb_array[vehicle_type,vehicle_depot,vehicle_no])-1)
                    # print('random_number_x:',random_number_x)
                    random_number_y = random.randint(0,len(perturb_array[vehicle_type,vehicle_depot,vehicle_no][0])-1)
                    # print('random_number_y:',random_number_y)
                    random_list=[-1,1]
                    if perturb_array[vehicle_type,vehicle_depot,vehicle_no][random_number_x][random_number_y][0]!=False:
                        d=random.choice(random_list)
                        # print('d:',d)
                        if random_number_y+d>=0 and random_number_y+d<len(perturb_array[vehicle_type,vehicle_depot,vehicle_no][0]) and perturb_array[vehicle_type,vehicle_depot,vehicle_no][random_number_x][random_number_y+d][0]==False: # The last condition makes it understand that the new column in the original row of the perturbation should be empty
                            print("Perturbed:: Row:",random_number_x,", Column:",random_number_y,", in the Direction:",d)
                            target_row=-1
                            for i in range(len(perturb_array[vehicle_type,vehicle_depot,vehicle_no])):
                                # print(perturb_array[vehicle_type,vehicle_depot,vehicle_no][i][random_number_y+d])
                                if perturb_array[vehicle_type,vehicle_depot,vehicle_no][i][random_number_y+d][0]!=False:
                                    target_row=i
                                    break
                            # route.clear()
                            # print('target_row:',target_row)
                            route=[]
                            for j in range(len(perturb_array[vehicle_type,vehicle_depot,vehicle_no][0])):
                                if j==random_number_y:
                                    route.append({perturb_array[vehicle_type,vehicle_depot,vehicle_no][target_row][random_number_y+d][0]:perturb_array[vehicle_type,vehicle_depot,vehicle_no][target_row][random_number_y+d][1]})
                                elif j==random_number_y+d:
                                    route.append({perturb_array[vehicle_type,vehicle_depot,vehicle_no][random_number_x][random_number_y][0]:perturb_array[vehicle_type,vehicle_depot,vehicle_no][random_number_x][random_number_y][1]})
                                else:
                                    for i in range(len(perturb_array[vehicle_type,vehicle_depot,vehicle_no])):
                                        if perturb_array[vehicle_type,vehicle_depot,vehicle_no][i][j][0]!=False:
                                            route.append({perturb_array[vehicle_type,vehicle_depot,vehicle_no][i][j][0]:perturb_array[vehicle_type,vehicle_depot,vehicle_no][i][j][1]})
                                            break
                            # print('route:',route)
                            if feasible(route,vehicle_type):
                                
                                nodes=[]
                                for item in route:
                                    if item==False:
                                        break
                                    for key in item:
                                        nodes.append(key)
                                print("\t Perturbation Successful")
                                
                                # print(route)
                                t=final_time(route,vehicle_type,vehicle_depot)
                                # print(t)
                                if t<temp:
                                    print('Route Vertices: ',nodes)
                                    print("Time Improvement: ",temp-t," seconds")
                                    temp=t
                                    perturb_array[vehicle_type,vehicle_depot,vehicle_no][target_row][random_number_y]=perturb_array[vehicle_type,vehicle_depot,vehicle_no][target_row][random_number_y+d]
                                    perturb_array[vehicle_type,vehicle_depot,vehicle_no][random_number_x][random_number_y+d]=perturb_array[vehicle_type,vehicle_depot,vehicle_no][random_number_x][random_number_y]
                                    perturb_array[vehicle_type,vehicle_depot,vehicle_no][random_number_x][random_number_y]=[False,False]
                                    perturb_array[vehicle_type,vehicle_depot,vehicle_no][target_row][random_number_y+d]=[False,False]
                                else:
                                    # We could accept sub-optimal solutions with low probability to break away from local optimas
                                    pass
                                
                    x=x+1
                max_T=max(max_T,temp)
                perturbed_route_time_mapping[vehicle_type,vehicle_depot,vehicle_no]=temp


#perturb_array[vehicle_type,vehicle_depot,vehicle_no]



end_time=time.time()



elapsed_time=end_time-start_time
print(f"Elapsed Time: {round(elapsed_time,3)} seconds")

print("max_T: ", round(max_T,5))
print("Maximum Route Time before any Perturbation: ", round(max_T_before_perturbation,5))


#list_of_routes


#perturb_array


for vehicle_type in Vehicles_Specifications_VehicleType_arr:
    for vehicle_depot in vehicle_depots_in_vehicle_type[vehicle_type]:
        for vehicle_no in range(1,vN[vehicle_depot,vehicle_type]+1):
            print("\n Vehicle ID: ", vehicle_type,",",vehicle_depot,",",vehicle_no)
            
            
            # The below Line prints the detailed Barcodes representing what is happening at each Node during each visit
            #print(list_of_routes[vehicle_type,vehicle_depot,vehicle_no])
            
            if list_of_routes[vehicle_type,vehicle_depot,vehicle_no]:
                print("\nOriginal Route Time: ",dynamic_time[vehicle_type,vehicle_depot,vehicle_no][0],", Final Vertex: ",dynamic_time[vehicle_type,vehicle_depot,vehicle_no][1])
                string_route="Original Route Starts:\n"
                for individual_route in list_of_routes[vehicle_type,vehicle_depot,vehicle_no]:
                    for i in individual_route.keys():
                        string_route+=i
                        string_route+="  -=>  "
                    #string_route+=" \n -=>  "        
                print(string_route[ : -5])
            
            if perturb_array[vehicle_type,vehicle_depot,vehicle_no]:
                string_perturbed_route="Perturbed Route Starts:\n"
                for j in range(len(perturb_array[vehicle_type,vehicle_depot,vehicle_no][0])): # Columns
                    for i in range(len(perturb_array[vehicle_type,vehicle_depot,vehicle_no])): # Rows
                        if perturb_array[vehicle_type,vehicle_depot,vehicle_no][i][j][0]!=False:
                            string_perturbed_route+=perturb_array[vehicle_type,vehicle_depot,vehicle_no][i][j][0]
                            string_perturbed_route+="  -=>  "
                            break
                print(string_perturbed_route[ : -5])
                print("Original Route Time: ",perturbed_route_time_mapping[(vehicle_type,vehicle_depot,vehicle_no)],"\n")


perturbed_route_time_mapping


# # Checking improvements, if any, after Perturbation:
# 
# Elapsed Time: 1.2102117538452148 seconds
# max_T:  1948.4438910523918
# Maximum Route Time before any Perturbation:  1948.4438910523918
# 
# # Single Perturbation is not good and requires further random moves, this can be proved by increasing the vehicle capacity to a very high value
# 
# # Also, create a Magnetic effect where the same Nodes on a route will try to come closer


range(len(perturb_array[vehicle_type,vehicle_depot,vehicle_no][0]))

if perturb_array[vehicle_type,vehicle_depot,vehicle_no]:
    print("kol")


#list_of_routes[vehicle_type,vehicle_depot,vehicle_no]
list_of_routes


print("\nOriginal Route Time: ",dynamic_time[vehicle_type,vehicle_depot,vehicle_no][0],", Final Vertex: ",dynamic_time[vehicle_type,vehicle_depot,vehicle_no][1])
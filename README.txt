# NDMVRP — Vehicle Routing Plugin for Natural Disaster Management

A QGIS plugin that generates optimised vehicle routing strategies for natural disaster management scenarios. Supports heterogeneous vehicle fleets, multi-cargo pickup/delivery operations, multimodal transport networks, and simultaneous/split node logistics.

---

## Requirements

- QGIS 3.0 or later
- Windows (primary supported platform)
- Internet connection on first run (for automatic dependency installation)

The plugin automatically installs missing Python packages (`openpyxl`, `lxml`, `Pillow`) on first activation. No manual setup is required on most machines.

---

## Installation

### Method 1 — Install from ZIP (Recommended)

> ⚠️ **Important:** If you downloaded the repository from GitHub using the "Download ZIP" button, the extracted folder will be named something like `NDMVRP-main`. You **must rename the ZIP file to `NDMVRP.zip` before installing**, otherwise QGIS cannot load the plugin.

**Steps:**

1. Download or obtain `NDMVRP.zip`
2. Rename it to `NDMVRP.zip` if it has any other name
3. Open QGIS
4. Go to **Plugins → Manage and Install Plugins → Install from ZIP**
5. Browse to `NDMVRP.zip` and click **Install Plugin**
6. On first activation, a dialog will appear saying packages are being installed — click OK and wait (~30 seconds)
7. The plugin icon will appear in the QGIS toolbar

### Method 2 — Manual Installation

1. Download and extract the ZIP
2. Rename the extracted folder to exactly `NDMVRP` (no hyphens, no suffixes)
3. Copy the `NDMVRP` folder into your QGIS plugins directory:
   ```
   C:\Users\<YourName>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\
   ```
4. Restart QGIS
5. Go to **Plugins → Manage and Install Plugins** and enable **NDMVRP**

---

## Manual Dependency Installation (if automatic install fails)

If you see a warning that automatic installation failed (usually due to missing administrator rights), open the **OSGeo4W Shell** as Administrator and run:

```bash
pip install --force-reinstall openpyxl lxml Pillow
```

Then restart QGIS.

---

## Usage

The plugin follows a strict sequential workflow. Each step must be completed before the next one becomes available.

### Step 1 — Add Vehicle Types

Click **+ ADD VEHICLE TYPES** and fill in the form for each vehicle type in your fleet:

| Field | Description |
|---|---|
| Description | Name or label for this vehicle type |
| Weight Capacity | Maximum cargo weight the vehicle can carry |
| Volume Capacity | Maximum cargo volume the vehicle can carry |
| Vehicle Network Compatibility | Which transport network this vehicle can use (1, 2, or 3) |
| Average Speed | Average travel speed of the vehicle |
| Return to Depot | Whether vehicles must return to their starting depot after completing routes |

Repeat for each vehicle type. All entries appear in the **Vehicle Details** table.

### Step 2 — Add Cargo Types

Click **+ ADD PICKUP CARGO** or **+ ADD DELIVERY CARGO** for each cargo type:

| Field | Description |
|---|---|
| Description | Name or label for this cargo type |
| Unit Weight | Weight per unit of this cargo |
| Unit Volume | Volume per unit of this cargo |

Pickup cargo is coded as `CC{n}P` and delivery cargo as `CC{n}D` automatically.

### Step 3 — Generate Compatibility Matrix

Click **SAVE VEHICLE & CARGO DATA**. A table will appear showing each vehicle type against each cargo type. Enter a compatibility value (default `1` = compatible) and loading/unloading time. Click **OK**.

This step also unlocks the location-adding buttons in the lower panel.

### Step 4 — Add Locations on the Map

Use the six location buttons to click points directly on the QGIS map canvas:

| Button | Code | Description |
|---|---|---|
| ADD DEPOT | `VD{n}` | Vehicle starting/ending depot |
| ADD WAREHOUSE | `WH{n}` | Intermediate storage point |
| ADD TRANSHIPMENT PORT | `TP{n}` | Cargo transfer between networks |
| ADD SIMULTANEOUS NODES | `NM{n}` | Nodes requiring simultaneous service |
| ADD RELIEF CENTER | `RC{n}` | Final delivery destination |
| ADD SPLIT NODE | `NP{n}` | Nodes where cargo can be split |

When you click a button, QGIS minimises and you click a point on the map. A form then appears asking for quantities and network compatibility for that location.

Use **CLEAR LOCATION TABLE** to remove all added locations and start over.

Use **RESET VRP** to clear all data (vehicles, cargo, compatibility, and locations) and start the entire workflow from scratch. A confirmation dialog will appear before any data is deleted.

### Step 5 — Save Input Data

Click **SAVE LOCATION AND CARGO DATA** to export all four input CSV files to a folder of your choice:

```
<output folder>/
  ├── 0 Vehicles.csv
  ├── 0 Cargo.csv
  ├── 1 Vehicle Cargo Compatibility and Loading Unloading Time.csv
  └── 1 Locations and PickUp Delivery details.csv
```

### Step 6 — Generate Distance Matrices

Click **GENERATE DISTANCE / TIME MATRICES**. You will be prompted to select the same output folder. The plugin computes shortest-path travel costs across all three networks and saves:

```
<output folder>/
  ├── Distance_Matrix_for_Network_1.csv
  ├── Distance_Matrix_for_Network_2.csv
  └── Distance_Matrix_for_Network_3.csv
```

The matrices are also added as layers in the QGIS map canvas.

### Step 7 — Run VRP

Click **RUN VRP**. You will be prompted to select the folder containing the input CSV files from Step 5 and 6. The heuristic solver (`PSR_GIP.py`) will run and produce results in an `OutPut/` subfolder:

```
<input folder>/
  └── OutPut/
        ├── MakeSPAN_OutPut_0.xlsx   ← best solution
        ├── MakeSPAN_OutPut_1.xlsx
        ├── MakeSPAN_OutPut_2.xlsx
        └── VIZ_MakeSPAN_OutPut_0/  ← route visualisations
```

The best solution is automatically loaded and visualised as a route layer in QGIS.

---

## Input Folder Structure

Before clicking **RUN VRP**, your input folder must contain all six CSV files:

```
<input folder>/
  ├── 0 Vehicles.csv
  ├── 0 Cargo.csv
  ├── 1 Vehicle Cargo Compatibility and Loading Unloading Time.csv
  ├── 1 Locations and PickUp Delivery details.csv
  ├── Distance_Matrix_for_Network_1.csv
  ├── Distance_Matrix_for_Network_2.csv
  └── Distance_Matrix_for_Network_3.csv
```

All six files are generated by Steps 5 and 6. If any are missing the heuristic will fail.

---

## Transport Networks

The plugin supports up to three separate transport networks (road, rail, waterway, etc.), each defined as a GeoPackage file located in the `Networks/` subfolder of the plugin directory:

```
NDMVRP/
  └── Networks/
        ├── Network_1.gpkg
        ├── Network_2.gpkg
        └── Network_3.gpkg
```

Replace these files with your own network GeoPackages before use. Each vehicle type can be assigned compatibility with one or more networks via the Vehicle Network Compatibility field in Step 1.

---

## Troubleshooting

**Plugin shows a plug icon instead of the NDMVRP icon**
The icon file `icon.png` is missing from the plugin folder. Ensure it is present at `NDMVRP/icon.png`.

**"No module named NDMVRP/NDMVRP" on installation**
The ZIP has a nested folder structure. Rename the ZIP to `NDMVRP.zip` before installing (see Installation section above).

**Run VRP opens a new QGIS window instead of running the solver**
This is caused by an old version of the plugin using `python3` instead of the correct interpreter. Update to the latest version.

**MakeSPAN_OutPut_0.xlsx not found after running VRP**
The heuristic solver crashed. Check the QGIS Python console for `--- Heuristic stderr ---` output which will show the exact error. The most common cause is a missing Python package — run `pip install --force-reinstall openpyxl lxml Pillow` in the OSGeo4W Shell.

**DLL load failed while importing _imaging**
Pillow's C extension is broken in the QGIS Python environment. Run in OSGeo4W Shell as Administrator:
```bash
pip install --force-reinstall Pillow
```

---

## Repository

[https://github.com/sid0dodo1/NdmVRP-problems](https://github.com/sid0dodo1/NdmVRP-problems)

---

## Author

Developed by AEM — ndmvrp@gmail.com

© 2023. Licensed under the GNU General Public License v2.

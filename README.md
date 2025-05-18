# Project Drive

This tool uses road data to identify 'fun' driving routes.

Fun is defined as:
- twisty roads (i.e.: with a high amount of curvature and bends)
- avoiding residential areas (i.e.: where there is the potential for more variable speeds)
- routes that do not have too few or too many crossing points, giving you equal chance to cruise and change direction

The tool creates 'loops' from a starting point of a desired length, returning you to your point of origin - ideal for a fun day out of exploring! It proposes 5 of the best routes in a variety of directions for you to choose from, and shows these on an interactive map. A data file is also produced with more details of each route, allowing you to choose whichever fits with your driving style.

Use the below instructions to get started. There is also the Jupyter notebook `src/notebook.ipynb` for those who are more curious about the inner workings of the tool!

## Overview

- [`src/parse_data.py`](src/parse_data.py): Downloads road network data for a specified area, enriches it with curvature information from a KML file, and saves the results.
- [`src/path_finder.py`](src/path_finder.py): Loads the enriched road network and finds, scores, and visualizes routes that maximize road curvature.

## Setup

1. **Install dependencies**  
   Make sure you have Python 3.8+ and install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

2. **Prepare curvature data**  
   Obtain a KML file containing curvature data for your region of interest. To do this, navigate to https://roadcurvature.com/ and go to to the relevant region's download page. It is suggested to use the c_300 files to get data for moderately twisty roads. This is however not necessary and you may not want to do this if you are planning routes longer than ~50km as it may increase compute time.
   This will download as a KMZ file (which is simply a zipped KML file) - use the kmz_unzip script to extract the KML file and save it in the data folder.

## Step 1: Fetch and Enrich Road Data

Use [`src/parse_data.py`](src/parse_data.py) to download and process the road network.

### Example usage

```sh
python src/parse_data.py \
  --lat 53.1701 \
  --lon -1.7816 \
  --dist 25000 \
  --kml data/great-britain.c_300.kml \
  --out data/
```

**Arguments:**
- `--lat` and `--lon`: Latitude and longitude of the center point.
- `--dist`: Search radius in meters (default: 25000).
- `--kml`: Path to the KML file with curvature data.
- `--out`: Output directory (default: `data/`).
- `--network`: Network type (default: `drive`).

**Outputs:**
- `data/edges_roads_with_curvature.gpkg`
- `data/nodes_roads_with_curvature.gpkg`
- GeoPackage files with curvature-enriched road data.

## Step 2: Find and Visualize High-Curvature Routes

Use [`src/path_finder.py`](src/path_finder.py) to find and visualize the best routes.

### Example usage

```sh
python src/path_finder.py \
  --edges data/edges_roads_with_curvature.gpkg \
  --nodes data/nodes_roads_with_curvature.gpkg \
  --start-lat 53.1701 \
  --start-lon -1.7816 \
  --target-distance 20000 \
  --output results/
```

**Arguments:**
- `--edges`: Path to the edges GeoPackage file.
- `--nodes`: Path to the nodes GeoPackage file.
- `--start-lat` and `--start-lon`: Start location for route search.
- `--target-distance`: Desired route length in meters.
- `--output`: Output directory for results.

**Outputs:**
- `results/results.html`: Interactive map with the top routes.
- `results/results.csv`: Route metrics and scores.

## Notes

- Ensure your KML curvature data covers the area you are analyzing.
- The scripts may take several minutes to run, depending on the area size and network complexity.

## Troubleshooting

- If you encounter CRS warnings, ensure all spatial data uses the same coordinate reference system (EPSG:4326).
- For Overpass API issues, check your internet connection or try again later.

## License

See [LICENSE](LICENSE) for details.

## Acknowledgements

- Road network data is sourced from [OpenStreetMap](https://www.openstreetmap.org/).
- Curvature data is provided by [roadcurvature.com](https://roadcurvature.com/).
- This project uses the following open source libraries:
  - [OSMnx](https://github.com/gboeing/osmnx)
  - [GeoPandas](https://geopandas.org/)
  - [NetworkX](https://networkx.org/)
  - [Folium](https://python-visualization.github.io/folium/)
  - [scikit-learn](https://scikit-learn.org/)
  - [tqdm](https://tqdm.github.io/)
- Please respect the licenses and terms of use for all data and software dependencies.
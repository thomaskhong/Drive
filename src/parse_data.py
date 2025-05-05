import osmnx as ox
import geopandas as gpd
from fastkml import kml
from shapely.geometry import shape
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def fetch_roads(place_name, network_type='drive'):
    print(f"Fetching road network for: {place_name}")
    G = ox.graph_from_place(place_name, network_type=network_type)
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    edges = edges.to_crs("EPSG:4326")
    return nodes, edges

def parse_kml_curvature(kml_path):
    print(f"Parsing curvature data from: {kml_path}")
    curvature_gdf = gpd.read_file(kml_path, driver='KML')
    curvature_gdf['curvature'] = [int(float(x[37:42])) for x in curvature_gdf["Description"]]
    return curvature_gdf

def join_curvature(edges, curvature_gdf):
    print("Joining curvature data to edges...")
    edges_with_curviness = gpd.sjoin_nearest(
        edges,
        curvature_gdf[['geometry', 'curvature']],
        how="left",
        max_distance=0.0005,  # ~50 meters
        distance_col="join_dist"
    )
    edges_with_curviness['curvature'] = edges_with_curviness['curvature'].fillna(0)
    edges_with_curviness.drop(columns=['join_dist', 'index_right'], inplace=True)
    edges_with_curviness = edges_with_curviness[~edges_with_curviness.index.duplicated(keep='first')]
    return edges_with_curviness

def save_outputs(edges, nodes, save_path, output_prefix="roads_with_curvature"):
    print(f"Saving data to {output_prefix}.csv and .geojson")
    edges.to_csv(f"{save_path}{output_prefix}.csv", index=False)
    edges.to_file(f"{save_path}edges_{output_prefix}.gpkg", layer='edges', driver='GPKG')
    nodes.to_file(f"{save_path}nodes_{output_prefix}.gpkg", layer='nodes', driver='GPKG')

def main():
    place = "Peak District"  # 🔁 Change this as needed
    kml_path = "data\great-britain.c_300.kml"               # 🔁 Replace with your file
    save_path = "data/"

    nodes, edges = fetch_roads(place)
    curvature_gdf = parse_kml_curvature(kml_path)
    enriched_edges = join_curvature(edges, curvature_gdf)
    save_outputs(enriched_edges, nodes.copy(), save_path)

if __name__ == "__main__":
    main()

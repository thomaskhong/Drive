import osmnx as ox
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd
import warnings
import argparse

warnings.filterwarnings("ignore", category=UserWarning)

def fetch_roads(center_point, dist_meters=1000, network_type='drive'):
    """
    Fetch road network data around a central coordinate within a specified distance.

    Args:
        center_point (tuple): (latitude, longitude)
        dist_meters (int): Search radius in meters
        network_type (str): Type of roads to retrieve (e.g., 'drive', 'walk')

    Returns:
        nodes, edges (GeoDataFrames)
    """
    print(f"Fetching road network within {dist_meters}m of {center_point}")
    ox.utils.settings.overpass_rate_limit = False
    ox.utils.settings.overpass_url = "https://overpass.kumi.systems/api"
    G = ox.graph_from_point(center_point, dist=dist_meters, network_type=network_type)
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

    parser = argparse.ArgumentParser(description="Fetch and enrich road data with curvature.")
    parser.add_argument('--lat', type=float, required=True, help='Latitude of the center point')
    parser.add_argument('--lon', type=float, required=True, help='Longitude of the center point')
    parser.add_argument('--dist', type=int, default=25000, help='Search radius in meters (default: 25000)')
    parser.add_argument('--kml', type=str, default='data/', required=True, help='Path to the KML file with curvature data')
    parser.add_argument('--out', type=str, default='data/', help='Output directory (default: data/)')
    parser.add_argument('--network', type=str, default='drive', help="Network type (e.g., 'drive', 'walk')")

    args = parser.parse_args()

    center_point = (args.lat, args.lon)
    dist_meters = args.dist
    kml_path = args.kml
    save_path = args.out
    network_type = args.network

    nodes, edges = fetch_roads(center_point, dist_meters, network_type)
    curvature_gdf = parse_kml_curvature(kml_path)
    enriched_edges = join_curvature(edges, curvature_gdf)
    save_outputs(enriched_edges, nodes.copy(), save_path)

if __name__ == "__main__":
    main()

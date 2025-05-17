import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
import argparse
from shapely.geometry import LineString
from tqdm import tqdm
from sklearn.cluster import KMeans
import folium
import matplotlib.pyplot as plt
from itertools import cycle

def normalize_vector(vec):
    vec = np.array(vec)
    return (vec - vec.min()) / (vec.max() - vec.min()) if vec.max() > vec.min() else np.zeros_like(vec)


def load_graph(edges_path, nodes_path):
    edges = gpd.read_file(edges_path, layer='edges')
    nodes = gpd.read_file(nodes_path, layer='nodes')
    edges.set_index(['u', 'v', 'key'], inplace=True)
    nodes.set_index(['osmid'], inplace=True)
    return ox.graph_from_gdfs(nodes, edges)


def calculate_outbound_paths(G, start_node, target_distance=2000, max_depth=50, top_k=2, max_paths=2000):
    stack = [([start_node], 0)]
    outbound_paths = []

    while stack and len(outbound_paths) < max_paths:
        new_stack = stack
        stack = []
        for option in new_stack:
            path, total_length = option
            current_node = path[-1]

            if total_length >= target_distance or len(path) > max_depth:
                outbound_paths.append(path)
                continue

            candidates = []

            for neighbor in G.successors(current_node):
                if neighbor in path:
                    continue  # avoid node re-use

                for key, data in G[current_node][neighbor].items():
                    length = data.get("length", 0)
                    curvature = data.get("curvature", 0)
                    score = (curvature+0.01) * length
                    candidates.append((score, neighbor, length))

            # Sort and take top-k
            candidates.sort(reverse=True)
            top_choices = candidates[:top_k]

            for _, next_node, edge_length in top_choices:
                new_path = path + [next_node]
                new_total_length = total_length + edge_length
                stack.append((new_path, new_total_length))

    return outbound_paths

def analyze_path_metrics(G, path, path_id, start_lat, start_lon, nodes):
    total_length = 0
    curvatures = []
    speedlims = []
    road_names = set()

    for u, v in zip(path[:-1], path[1:]):
        if G.has_edge(u, v):
            # For MultiDiGraph, get all edges between u and v
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                # Choose edge with maximum curvature * length
                best_key = max(
                    edge_data,
                    key=lambda k: edge_data[k].get('curvature', 0) * edge_data[k].get('length', 0)
                )
                data = edge_data[best_key]

                total_length += data.get("length", 0)
                curvatures.append(data.get("curvature", 0) / data.get("length", 1))

                # Get speed limit
                speed_limit = data.get("maxspeed")
                if isinstance(speed_limit, str):
                    try:
                        speedlims.append(int(speed_limit.removesuffix(' mph')) / data.get("length", 1))
                    except:
                        speedlims.append(30 / data.get("length", 1))
                elif speed_limit is None:
                    speedlims.append(30)

                # Add road name(s) to set (can be list or string)
                name = data.get("name")
                if isinstance(name, list):
                    road_names.update(name)
                elif isinstance(name, str):
                    road_names.add(name)
                elif name is None:
                    road_names.add(data.get("osmid"))
        else:
            print(f"Warning: no edge from {u} to {v}")

    avg_curvature = sum(curvatures) / len(curvatures) if curvatures else 0
    num_unique_roads = len(road_names) / len(path)
    avg_speed = sum(speedlims) / len(speedlims)

    # calculate bearing relative to start point

    node_centroid = (np.average(nodes[nodes.index.isin(path)]['y']), np.average(nodes[nodes.index.isin(path)]['x']))
    bearing = np.atan2((node_centroid[0]-start_lat), (node_centroid[1]-start_lon))

    return path_id, total_length, avg_curvature, num_unique_roads, avg_speed, bearing, path

def calculate_scores(G, input_paths, start_lat, start_lon, nodes):
    
    results = []
    for idx, path in enumerate(input_paths):
        results.append(analyze_path_metrics(G, path, idx, start_lat, start_lon, nodes))
    
    path_metrics = pd.DataFrame(results, columns=['path_id', 'total_length', 'avg_curvature', 'unique_road_num', 'avg_speed', 'bearing', 'path_nodes'])
    
    curvature_vector_norm = normalize_vector(path_metrics['avg_curvature'].to_list())
    roadnumber_vector_norm = normalize_vector(path_metrics['unique_road_num'].to_list())
    speed_vector_norm = normalize_vector(path_metrics['avg_speed'].to_list())
    score_vector = 2*curvature_vector_norm - roadnumber_vector_norm + speed_vector_norm

    path_metrics['score'] = score_vector

    return path_metrics


def remove_similar_paths(path_scores, similarity_threshold=0.5):
    # Convert lists to sets for faster operations
    path_scores['path_nodes_set'] = path_scores['path_nodes'].apply(set)

    # Sort by score descending
    path_scores = path_scores.sort_values('score', ascending=False).reset_index(drop=True)

    # Store accepted list indices
    accepted_indices = []
    accepted_sets = []

    for i, row in tqdm(path_scores.iterrows(), total=len(path_scores), disable=True):
        current_set = row['path_nodes_set']
        is_similar = False
        for other_set in accepted_sets:
            intersection = len(current_set & other_set)
            union = len(current_set | other_set)
            similarity = intersection / union
            if similarity >= similarity_threshold:
                is_similar = True
                break
        if not is_similar:
            accepted_indices.append(i)
            accepted_sets.append(current_set)

    # Result
    filtered_paths = path_scores.loc[accepted_indices].reset_index(drop=True)
    return filtered_paths


def cluster_paths(path_scores, n_clusters=4):
    """
    Cluster route bearings using polar coordinates and scores.
    
    bearings: list of bearing values in degrees
    scores: list of route scores

    Returns:
        cluster_labels: array of cluster assignments for each route
    """
    # Convert bearings to radians and use unit circle projection
    radians = path_scores['bearing'].to_list()
    x = np.cos(radians)
    y = np.sin(radians)
    
    # Combine with score to cluster in 3D space
    X = np.column_stack([x, y, path_scores['score'].to_list()])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    path_scores['labels'] = labels

    return path_scores

def select_top_paths_by_cluster(filtered_results: pd.DataFrame, top_n=1) -> pd.DataFrame:
    """
    Selects top N scoring paths from each cluster in a DataFrame.

    Returns:
        A DataFrame containing the top N paths from each cluster, retaining all columns.
    """
    top_paths_df = (
        filtered_results
        .sort_values(['labels', 'score'], ascending=[True, False])  # sort by cluster, then score descending
        .groupby('labels', group_keys=False)
        .head(top_n)
    )
    return top_paths_df.reset_index(drop=True)


def get_edges_from_path(G, node_path):
    """
    Given a list of node IDs representing a path, return a list of (u, v, key) edges.
    Assumes G is a MultiDiGraph where multiple edges may exist between nodes.
    """
    edges = []
    for i in range(len(node_path) - 1):
        u = node_path[i]
        v = node_path[i + 1]

        if G.has_edge(u, v):
            # Choose the first matching edge key if multiple exist
            key = list(G[u][v].keys())[0]
            edges.append((u, v, key))
        else:
            print(f"Warning: No edge from {u} to {v} in graph.")
    return edges


def build_return_graph(G, outbound_edges, curvature_weight=1.0, reuse_penalty=10000):
    G_ret = G.copy()
    for u, v, k, data in G_ret.edges(keys=True, data=True):
        base_cost = 1  # or use length, if you want to bias short paths
        curvature = data.get("curvature", 0)
        reuse = (u, v, k) in outbound_edges or (v, u, k) in outbound_edges

        cost = base_cost - curvature_weight * curvature
        if reuse:
            cost += reuse_penalty

        G_ret[u][v][k]["weight"] = max(cost, 0.01)  # prevent zero or negative weights

    return G_ret


def create_full_loops(G, best_outbound_paths, start_node):
    full_loops = []
    for out_path in best_outbound_paths['path_nodes'].to_list():
        outbound_edges = get_edges_from_path(G, out_path)
        G_return = build_return_graph(G, outbound_edges)
        return_path = nx.shortest_path(G_return, source=out_path[-1], target=start_node, weight="weight")
        full_loops.append(out_path + return_path[1:])  # skip duplicate end/start node
    
    best_outbound_paths['path_nodes'] = full_loops
    return best_outbound_paths


def plot_outputs(G, proposed_paths, start_lat, start_lon, output_path):
    # --- Create Folium map ---
    m = folium.Map(location=[start_lat, start_lon], zoom_start=11)

    # Plot start
    folium.Marker([start_lat, start_lon], popup="Start", icon=folium.Icon(color='gray')).add_to(m)

    def get_color(val, vmin, vmax):
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('jet')
        rgba = cmap(norm(val))
        return f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})'

    colors = cycle(['red', 'blue', 'green', 'purple', 'orange'])
    path_ids = proposed_paths['path_id'].values.tolist()
    for path_num, path in enumerate(proposed_paths['path_nodes'].values.tolist()):

        path_edges = []
        path_color = next(colors)

        for u, v in zip(path[:-1], path[1:]):
            # Get all parallel edges between u and v (MultiDiGraph may have >1)
            edge = G.get_edge_data(u, v)
            
            if edge:
                # Select the shortest edge (or first) if multiple exist
                edge_data = min(edge.values(), key=lambda d: d.get('length', float('inf')))
                path_edges.append(edge_data)
                curvature = edge_data.get('curvature', 0)

        coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path]

        # Add edges to the map
        for row in path_edges:
            if isinstance(row['geometry'], list):
                # Skip any weird geometry
                continue
            folium.PolyLine(locations=[(lat, lon) for lon, lat in row['geometry'].coords],
                            color=path_color,
                            weight=5).add_to(m)

        # Plot helper markers
        marker_node = path[len(path)//2]
        folium.map.Marker([G.nodes[marker_node]['y'], G.nodes[marker_node]['x']], popup="Path Number: "+str(path_ids[path_num]), icon=folium.Icon(color=path_color)).add_to(m)

    # Show map
    m.save(output_path+"results.html")

def save_output(proposed_paths: pd.DataFrame, output_path):
    proposed_paths.to_csv(output_path+"results.csv")

def main():
    parser = argparse.ArgumentParser(description="Find high-curvature routes using a greedy search.")
    parser.add_argument('--edges', required=True, default='data/edges_roads_with_curvature.gpkg', help='Path to edges GeoPackage')
    parser.add_argument('--nodes', required=True, default='data/nodes_roads_with_curvature.gpkg', help='Path to nodes GeoPackage')
    parser.add_argument('--start-lat', type=float, required=True, help='Start latitude')
    parser.add_argument('--start-lon', type=float, required=True, help='Start longitude')
    parser.add_argument('--target-distance', type=float, default=2000, help='Target path length in meters')
    parser.add_argument('--output', required=True, help='Output file path')

    args = parser.parse_args()

    print('>> Preparing graph...')
    G = load_graph(args.edges, args.nodes)
    start_node = ox.nearest_nodes(G, args.start_lon, args.start_lat)
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    print('>> Calculating outbound paths...')
    outbound_paths = calculate_outbound_paths(G, start_node, target_distance=(args.target_distance)/2, max_paths=1500, max_depth=30)
    print('   ', {len(outbound_paths)}, 'paths found')
    print('>> Scoring outbound paths...')
    outbound_paths = calculate_scores(G, outbound_paths, args.start_lon, args.start_lat, nodes=nodes)
    print('>> Removing similar paths...')
    filtered_outbound_paths = remove_similar_paths(outbound_paths, similarity_threshold=0.5)
    print('>> Clustering results and selecting best outbound options...')
    clustered_outbound_paths = cluster_paths(filtered_outbound_paths, n_clusters=4)
    best_outbound_paths = select_top_paths_by_cluster(clustered_outbound_paths, top_n=5)
    print('>> Plotting return journeys for best outbound path candidates...')
    completed_paths = create_full_loops(G, best_outbound_paths, start_node)
    print('>> Calculating full journey scores and saving best options...')
    completed_paths = calculate_scores(G, completed_paths['path_nodes'].to_list(), args.start_lon, args.start_lat, nodes=nodes)
    proposed_paths = completed_paths.sort_values(by='score', ascending=False).head(5)
    plot_outputs(G, proposed_paths, args.start_lat, args.start_lon, args.output)
    save_output(proposed_paths, args.output)
    print('>> Results complete. View map in the results.html, and see route metrics in results.csv files. These have been saved to the output directory of your choice.')


if __name__ == '__main__':
    main()

"""NetworkX representation of nearby charging stations."""
import networkx as nx
import pandas as pd
from src.utils.geo import haversine_km

def build_station_graph(stations: pd.DataFrame, edge_radius_km: float=25.0) -> nx.Graph:
    graph=nx.Graph()
    for row in stations.itertuples(): graph.add_node(str(row.station_id), **row._asdict())
    rows=list(stations.itertuples())
    for i,left in enumerate(rows):
        for right in rows[i+1:]:
            distance=haversine_km(left.latitude,left.longitude,right.latitude,right.longitude)
            if distance<=edge_radius_km: graph.add_edge(str(left.station_id),str(right.station_id),distance_km=distance)
    return graph

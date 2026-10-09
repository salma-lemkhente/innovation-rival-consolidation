import networkx as nx
import pandas as pd

from projlib import config, io

rp = pd.read_parquet(config.out("07_panel", "rival_pair_year.parquet"), columns=["focal", "rival", "year"]).drop_duplicates()
w = rp.groupby(["focal", "rival"]).size().rename("w").reset_index()
G = nx.Graph()
for a, b, k in w.itertuples(index=False):
    if a == b:
        continue
    if G.has_edge(a, b):
        G[a][b]["weight"] += k
    else:
        G.add_edge(a, b, weight=k)
print(f"graph: {G.number_of_nodes():,} entities, {G.number_of_edges():,} edges, {nx.number_connected_components(G):,} components")
comms = nx.community.louvain_communities(G, weight="weight", seed=7)
rows = [(n, i, len(c)) for i, c in enumerate(sorted(comms, key=len, reverse=True)) for n in c]
out = pd.DataFrame(rows, columns=["focal", "comm_id", "comm_size"])
sizes = out.drop_duplicates("comm_id")["comm_size"]
print(f"communities: {len(sizes):,}; largest {sizes.max():,}; median {sizes.median():.0f}; with at least 10 entities {(sizes >= 10).sum():,}; entities in communities of at least 10: {out['comm_size'].ge(10).mean():.1%}")
io.write(out, config.out("08_estimation", "rival_communities.parquet"), key=["focal"])
print("written")

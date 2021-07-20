import numpy as np
import pandas as pd
import uclasm

# --

w_nodelist, w_channels, w_adjs = uclasm.load_combo(
    "world.csv",
    node_vs_edge_col=0,
    node_str="v",
    src_col=1,
    dst_col=2,
    channel_col=3,
    node_col=1,
    label_col=2,
    header=0)

t_nodelist, t_channels, t_adjs = uclasm.load_combo(
    "tmplt.csv",
    node_vs_edge_col=0,
    node_str="v",
    src_col=1,
    dst_col=2,
    channel_col=3,
    node_col=1,
    label_col=2,
    header=0)

# --

w_adjs = [w_adjs[i] for i in np.argsort(w_channels)]
t_adjs = [t_adjs[i] for i in np.argsort(t_channels)]

channels = np.sort(w_channels).astype(int)

# Use the same graph data for both template and world graphs
tmplt = uclasm.Graph(t_nodelist.node, channels, t_adjs, labels=t_nodelist.label)
world = uclasm.Graph(w_nodelist.node, channels, w_adjs, labels=w_nodelist.label)

# --

tmplt, world, candidates = uclasm.run_filters(
  tmplt, 
  world, 
  filters=uclasm.all_filters, 
  verbose=True
)

n_isos = uclasm.count_isomorphisms(
  tmplt,
  world,
  candidates=candidates,
  verbose=True
)

isos = uclasm.find_isomorphisms(tmplt, world, candidates=candidates, verbose=False)
isos = pd.DataFrame(isos)

isos.to_csv('isos.tsv', sep='\t', index=False)

print('n_isos', n_isos)
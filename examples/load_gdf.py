import io
import json
import numpy as np
import pandas as pd

# --
# GDF Helpers

type_lookup = {
  "VARCHAR" : str,
  "INT"     : pd.Int64Dtype(),
  "DOUBLE"  : pd.Float64Dtype(),
}

def _gdf_parsedef(x):
  x = x.split('>')[1].split(',')
  x = dict([xx.replace(':', '_').split() for xx in x])
  
  for k,v in x.items():
    x[k] = type_lookup[v]
  
  return x

def _gdf_to_pandas(x, schema):
  return pd.read_csv(
    io.StringIO('\n'.join(x)),
    names     = list(schema.keys()),
    dtype     = schema,
    na_values = ['%NA%', '%NULL%'],
  )


def read_gdf(inpath):
  x = open(inpath).read().splitlines()
  
  for i, xx in enumerate(x):
    if xx[:8] == 'nodedef>':
      nodedef_idx = i
    if xx[:8] == 'edgedef>':
      edgedef_idx = i
  
  nodedef = _gdf_parsedef(x[nodedef_idx])
  df_node = _gdf_to_pandas(x[nodedef_idx + 1:edgedef_idx], nodedef)
  
  edgedef = _gdf_parsedef(x[edgedef_idx])
  df_edge = _gdf_to_pandas(x[edgedef_idx + 1:], edgedef)
  
  return df_node, df_edge

# --
# Template helpers

def _unlist(x):
  assert x.apply(len).max() == 1
  return x.apply(lambda x: x[0])

def _clean_tmplt(x):
  df         = pd.DataFrame(x)
  df.columns = [c.replace(':', '_') for c in df.columns]
  
  if 'rdf_type' in df.columns:
    df.rdf_type = _unlist(df.rdf_type)
  
  if 'node1' in df.columns:
    df.node1 = _unlist(df.node1)
  
  if 'node2' in df.columns:
    df.node2 = _unlist(df.node2)
  
  if 'argument' in df.columns:
    df.argument = df.argument.apply(lambda x: x if isinstance(x, list) else [None])
    assert df.argument.apply(len).max() == 1
    df.argument = df.argument.apply(lambda x: x[0])
  
  return df

def read_tmplt(inpath):
  tmplt = json.load(open(inpath))
  
  df_node = _clean_tmplt(tmplt['nodedef'])
  df_edge = _clean_tmplt(tmplt['edgedef'])
  
  return df_node, df_edge

# --
# Load world

w_path = '../maa_data/V2_2_HRL_alignment.gdf'
w_node, w_edge = read_gdf(w_path)

# >>
w_edge = w_edge.head(5_000)
# <<

# --
# Load template

t_path = '/home/ubuntu/projects/umdMAA/python/test_tmplt.json'
t_node, t_edge = read_tmplt(t_path)

# --
# Pre-filter

# tmp      = t_node[t_node.rdf_type == 'Transaction.TransferMoney']
# tmp      = tmp[tmp.numericValue.notnull()]
# min_val  = tmp.numericValue.apply(lambda x: x['minValue']).min()
# max_val  = tmp.numericValue.apply(lambda x: x['maxValue']).max()

# sel      = w_node.rdf_type == 'Transaction.TransferMoney'
# tmp      = w_node[sel].numericValue
# sel2     = tmp.isnull() | ((tmp >= min_val) & (tmp <= max_val))
# w_node   = w_node[~sel | sel2]

# w_unodes = set(w_node.name)
# sel      = w_edge.node1.isin(w_unodes) & w_edge.node2.isin(w_unodes)
# w_edge   = w_edge[sel]

w_unodes = set(w_edge.node1) | set(w_edge.node2)
sel      = w_node.name.isin(w_unodes)
w_node   = w_node[sel]

print('n_node', w_node.shape[0])
print('n_edge', w_edge.shape[0])

# --
# Format

# tmplt
t_node = t_node[['template_id', 'rdf_type']]
t_node.columns = ('node_id', 'node_type')

t_edge = t_edge[['node1', 'node2', 'rdf_type', 'argument']]
t_edge = t_edge[t_edge.rdf_type.isin(['Transaction.TransferMoney'])]
t_edge['channel'] = t_edge.rdf_type + '/' + t_edge.argument
t_edge = t_edge[['node1', 'node2', 'channel']]

# world
w_node = w_node[['name', 'rdf_type']]
w_node.columns = ('node_id', 'node_type')

w_edge = w_edge[['node1', 'node2', 'rdf_type', 'argument']]
w_edge = w_edge[w_edge.rdf_type.isin(['Transaction.TransferMoney'])]
w_edge['channel'] = w_edge.rdf_type + '/' + w_edge.argument
w_edge = w_edge[['node1', 'node2', 'channel']]

w_edge = w_edge[w_edge.channel.isin(set(t_edge.channel))] # !! filter

print('n_node', w_node.shape[0])
print('n_edge', w_edge.shape[0])

# --
# Remap labels

node_types  = w_node.node_type.unique()
node_lookup = dict(zip(node_types, range(len(node_types))))
w_node.node_type = w_node.node_type.apply(lambda x: node_lookup[x])
t_node.node_type = t_node.node_type.apply(lambda x: node_lookup[x])

edge_types     = w_edge.channel.unique()
edge_lookup    = dict(zip(edge_types, range(len(edge_types))))
w_edge.channel = w_edge.channel.apply(lambda x: edge_lookup[x])
t_edge.channel = t_edge.channel.apply(lambda x: edge_lookup[x])

# --

def write_combo(node, edge, fname):
  node = node.copy()
  edge = edge.copy()
  
  with open(fname, 'w') as f:
    print('Node/Edge,NodeId/Source,Label/Destination,Channel', file=f)
  
  node['_type'] = 'v'
  node = node[['_type', 'node_id', 'node_type']]
  node.node_type = node.node_type.astype(int)
  node.to_csv(open(fname, 'a'), header=None, index=False)
  
  edge['_type'] = 'e'
  edge = edge[['_type', 'node1', 'node2', 'channel']]
  edge.channel = edge.channel.astype(int)
  edge.to_csv(open(fname, 'a'), header=None, index=False)


write_combo(t_node, t_edge, 'tmplt.csv')
write_combo(w_node, w_edge, 'world.csv')
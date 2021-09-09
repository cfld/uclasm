#!/usr/bin/env python

"""
  run_maa.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

import uclasm

# --
# IO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',  type=str, default='/home/ubuntu/projects/umdMAA/python/data')
    parser.add_argument('--world', type=str, default='V2_2_HRL_alignment')
    parser.add_argument('--tmplt', type=str, default='V4_1sender-2payees1city')
    parser.add_argument('--mode',  type=str, default='no_sc')
    
    args = parser.parse_args()
    assert args.mode == 'no_sc'
    args.indir  = os.path.join(args.root, args.mode, args.world, args.tmplt)
    args.outdir = os.path.join('output', args.world, args.tmplt)
    
    return args

args = parse_args()
os.makedirs(os.path.dirname(args.outdir), exist_ok=True)

_load_combo_kwargs = {
    "node_vs_edge_col" : 0,
    "node_str"         : "v",
    "src_col"          : 1,
    "dst_col"          : 2,
    "channel_col"      : 3,
    "node_col"         : 1,
    "label_col"        : 2,
    "header"           : 0,
}

w_nodelist, w_channels, w_adjs = uclasm.load_combo(os.path.join(args.indir, "uc_world.csv"), **_load_combo_kwargs)
t_nodelist, t_channels, t_adjs = uclasm.load_combo(os.path.join(args.indir, "uc_tmplt.csv"), **_load_combo_kwargs)

# --
# Form Graph

w_adjs = [w_adjs[i] for i in np.argsort(w_channels)]
t_adjs = [t_adjs[i] for i in np.argsort(t_channels)]

channels = np.sort(w_channels).astype(int)

tmplt = uclasm.Graph(t_nodelist.node, channels, t_adjs, labels=t_nodelist.label)
world = uclasm.Graph(w_nodelist.node, channels, w_adjs, labels=w_nodelist.label)

# --
# Run

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

# --
# Save

print({'n_isos' : int(n_isos)}, file=sys.stderr)
isos.to_csv(args.outdir, sep='\t', index=False, header=None)

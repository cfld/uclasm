"""Example usage of the uclasm package for finding subgraph isomorphisms."""

import sys
sys.path.append('/home/ebarnett/uclasm')
print(sys.path)
import uclasm

tmplt = uclasm.load_edgelist("template.csv",
                             file_source_col="Source",
                             file_target_col="Target",
                             file_channel_col="eType")

world = uclasm.load_edgelist("world.csv",
                             file_source_col="Source",
                             file_target_col="Target",
                             file_channel_col="eType")

smp = uclasm.MatchingProblem(tmplt, world)

uclasm.matching.local_cost_bound.nodewise(smp)
uclasm.matching.global_cost_bound.from_local_bounds(smp)

print(smp)


import uclasm
from load_data_combo import tmplt, world

# Count
tmplt, world, candidates = uclasm.run_filters(tmplt, world, filters=uclasm.all_filters, verbose=True)
n_isomorphisms = uclasm.count_isomorphisms(tmplt, world, candidates=candidates, verbose=False)

# Find
isos = uclasm.find_isomorphisms(tmplt, world, candidates=candidates, verbose=False)
isos

print("\nFound", n_isomorphisms, "isomorphisms")

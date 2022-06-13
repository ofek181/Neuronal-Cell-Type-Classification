from allensdk.core.cell_types_cache import CellTypesCache
import pandas as pd

ctc = CellTypesCache()

# download all electrophysiology features for all cells
ephys_features = ctc.get_ephys_features()
ef_df = pd.DataFrame(ephys_features)

print("Ephys. features available for %d cells" % len(ef_df))

# filter down to a specific cell
specimen_id = 464212183
cell_ephys_features = ef_df[ef_df['specimen_id']== specimen_id]

print(cell_ephys_features)


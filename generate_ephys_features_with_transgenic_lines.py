from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")

dir_path = os.path.dirname(os.path.realpath(__file__))
dataframe_path = dir_path + '/data/dataframe'
dataframe_path_mouse = dataframe_path + '/mouse'


class Downloader:
    def __init__(self) -> None:
        """
        :param human: defines whether to take human cells or mouse cells for the experiment.
        """
        self.ctc = CellTypesCache()
        self.cells = self.ctc.get_cells(species=[CellTypesApi.MOUSE])
        self.esf = EphysSweepFeatureExtractor

    def save_ephys_data(self, ephys_data: dict, title: str) -> None:
        """
        :param ephys_data: the cell electrophysiology data.
        :param title: title of the file.
        :return: saves data to path in pkl and csv formats.
        """
        df = pd.DataFrame(data=ephys_data).transpose()
        df['layer'] = df['layer'].replace(['6a', '6b'], 6)
        df['layer'] = df['layer'].replace('2/3', 2)
        df['layer'] = df['layer'].astype('int')
        df = df[df['dendrite_type'].isin(['spiny', 'aspiny'])]
        df['file_name'] = df.index
        df.to_csv(os.path.join(dataframe_path_mouse, title), index=False)

    @staticmethod
    def get_ephys_features(sweep_data: dict) -> dict:
        """
        :param sweep_data: sweep data for cell sweep.
        :return: analysis of ephys features.
        """
        index_range = sweep_data["index_range"]
        i = sweep_data["stimulus"][0:index_range[1] + 1]
        v = sweep_data["response"][0:index_range[1] + 1]
        i *= 1e12
        v *= 1e3
        sampling_rate = sweep_data["sampling_rate"]
        t = np.arange(0, len(v)) * (1.0 / sampling_rate)
        result = {}
        sweep_ext = EphysSweepFeatureExtractor(t=t, v=v, i=i)
        sweep_ext.process_spikes()
        for key in sweep_ext.spike_feature_keys():
            try:
                result['mean_' + key] = np.mean(sweep_ext.spike_feature(key))
            except TypeError:
                continue
        return result

    @staticmethod
    def pipeline_checks(cell) -> bool:
        """
        :param cell: metadata about the cell
        :return: True whether to continue using the cell based on the described tests, else False
        """
        # take only Cre reporters which are positive
        if cell['reporter_status'] != 'positive':
            return False

        # merge transgenic classes based on transcriptomic cell types according to previous reports
        # Tasic et al., 2016, 2018
        # merge Nr5a1-Cre, Scnn1a-Tg2-Cre, Scnn1a-Tg3-Cre to one class, classified as layer 4 excitatory neurons
        if cell['transgenic_line'] == 'Nr5a1-Cre':
            cell['transgenic_line'] = 'Nr5a1-Scnn1a'
        if cell['transgenic_line'] == 'Scnn1a-Tg2-Cre':
            cell['transgenic_line'] = 'Nr5a1-Scnn1a'
        if cell['transgenic_line'] == 'Scnn1a-Tg3-Cre':
            cell['transgenic_line'] = 'Nr5a1-Scnn1a'
        # merge Cux2-CreERT2 and Slc17a6-IRES-CRE for the same reasons.
        if cell['transgenic_line'] == 'Cux2-CreERT2':
            cell['transgenic_line'] = 'Cux2-Slc17'
        if cell['transgenic_line'] == 'Slc17a6-IRES-CRE':
            cell['transgenic_line'] = 'Cux2-Slc17'
        # merge Chat-IRES-Cre-neo and Vip-IRES-Cre for the same reasons.
        if cell['transgenic_line'] == 'Chat-IRES-Cre-neo':
            cell['transgenic_line'] = 'Chat-Vip'
        if cell['transgenic_line'] == 'Vip-IRES-Cre':
            cell['transgenic_line'] = 'Chat-Vip'

        # exclude transgenic lines which are not correlated with their dendrite type
        # spiny_lines = ['Ctgf-T2A-dgCre', 'Cux2-Slc17', 'Nr5a1-Scnn1a',
        #                'Ntsr1-Cre_GN220', 'Rbp4-Cre_KL100', 'Rorb-IRES2-Cre']
        # aspiny_lines = ['Ndnf-IRES2-dgCre', 'Pvalb-IRES-Cre', 'Sst-IRES-Cre',
        #                 'Chat-Vip', 'Htr3a-Cre_NO152']
        spiny_lines = ['Ctgf-T2A-dgCre', 'Cux2-Slc17', 'Nr5a1-Scnn1a',
                       'Ntsr1-Cre_GN220', 'Rbp4-Cre_KL100', 'Rorb-IRES2-Cre']
        aspiny_lines = ['Ndnf-IRES2-dgCre', 'Pvalb-IRES-Cre', 'Sst-IRES-Cre',
                        'Chat-Vip', 'Htr3a-Cre_NO152']

        # spiny dendritic types correspond to excitatory pyramidal stellate neurons.
        if cell['dendrite_type'] == 'spiny' and cell['transgenic_line'] not in spiny_lines:
            return False
        # aspiny and sparsely spiny dendritic types correspond to inhibitory interneurons
        if cell['dendrite_type'] == 'sparsely spiny' and cell['transgenic_line'] not in aspiny_lines:
            return False
        if cell['dendrite_type'] == 'aspiny' and cell['transgenic_line'] not in aspiny_lines:
            return False

        return True

    def generate_data(self):
        """
        :return: creates the cell ephys db and saves data into known paths with different formats.
        """
        inhibitory_db = {}
        excitatory_db = {}
        for ind, cell in tqdm(enumerate(self.cells)):
            cell_id = cell['id']
            data_set = self.ctc.get_ephys_data(cell_id)
            sweeps = self.ctc.get_ephys_sweeps(cell_id)
            noise_sweep_number = [x['sweep_number'] for x in sweeps
                                  if x['stimulus_name'] in ['Noise 1', 'Noise 2']
                                  and x['num_spikes'] is not None
                                  and x['num_spikes'] > 10]
            if not noise_sweep_number:
                continue

            try:  # Make sure ephys file is not corrupted
                sweep_data = data_set.get_sweep(noise_sweep_number[0])
                ephys_feats = self.get_ephys_features(sweep_data)
            except:
                corrupted_filename = self.ctc.get_cache_path('', 'EPHYS_DATA', cell_id)
                os.remove(corrupted_filename)
                data_set = self.ctc.get_ephys_data(cell_id)

            for sweep_num in [noise_sweep_number[0]]:
                this_cell_id = '{}_{}'.format(cell_id, sweep_num)
                sweep_data = data_set.get_sweep(sweep_num)
                ephys_feats = self.get_ephys_features(sweep_data)
                if self.pipeline_checks(cell):
                    # aspiny and sparsely spiny dendritic types correspond to inhibitory interneurons
                    if cell['dendrite_type'] == 'aspiny' or cell['dendrite_type'] == 'sparsely spiny':
                        inhibitory_db[this_cell_id] = {**{'transgenic_line': cell['transgenic_line'],
                                                          'dendrite_type': cell['dendrite_type'],
                                                          'layer': cell['structure_layer_name']}, **ephys_feats}
                    # spiny dendritic types correspond to excitatory pyramidal stellate neurons.
                    if cell['dendrite_type'] == 'spiny':
                        excitatory_db[this_cell_id] = {**{'transgenic_line': cell['transgenic_line'],
                                                          'dendrite_type': cell['dendrite_type'],
                                                          'layer': cell['structure_layer_name']}, **ephys_feats}

        self.save_ephys_data(inhibitory_db, 'inhibitory_transgenic_data.csv')
        self.save_ephys_data(excitatory_db, 'excitatory_transgenic_data.csv')


if __name__ == '__main__':
    downloader = Downloader()
    downloader.generate_data()

#

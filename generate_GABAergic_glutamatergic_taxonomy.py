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

        # Cre lines that have similar characteristics are kept together
        # Zeng, Hongkui, and Joshua R. Sanes.
        # "Neuronal cell-type classification: challenges, opportunities and the path forward."
        # Nature Reviews Neuroscience 18.9 (2017): 530-546.
        # Tasic, Bosiljka, et al.
        # "Adult mouse cortical cell taxonomy revealed by single cell transcriptomics."
        # Nature neuroscience 19.2 (2016): 335-346.
        gabaergic = {'Pvalb-IRES-Cre':                    'GABAergic: Pvalb+',
                     'Slc32a1-T2A-FlpO|Vipr2-IRES2-Cre':  'GABAergic: Pvalb+',
                     'Nkx2-1-CreERT2':                    'GABAergic: Pvalb+',
                     'Ndnf-IRES2-dgCre':                  'GABAergic: Htr3a+|Vip-',
                     'Gad2-IRES-Cre':                     'GABAergic: Htr3a+|Vip-',
                     'Htr3a-Cre_NO152':                   'GABAergic: Htr3a+|Vip-',
                     'Sst-IRES-Cre':                      'GABAergic: Sst+',
                     'Nos1-CreERT2':                      'GABAergic: Sst+',
                     'Nos1-CreERT2|Sst-IRES-FlpO':        'GABAergic: Sst+',
                     'Oxtr-T2A-Cre':                      'GABAergic: Sst+',
                     'Chrna2-Cre_OE25':                   'GABAergic: Sst+',
                     'Chat-IRES-Cre-neo':                 'GABAergic: Vip+',
                     'Vip-IRES-Cre':                      'GABAergic: Vip+'}
        glutamatergic = {'Ctgf-T2A-dgCre':                'Glutamatergic',
                         'Tlx3-Cre_PL56':                 'Glutamatergic',
                         'Sim1-Cre_KJ18':                 'Glutamatergic',
                         'Glt25d2-Cre_NF107':             'Glutamatergic',
                         'Ntsr1-Cre_GN220':               'Glutamatergic',
                         'Esr2-IRES2-Cre':                'Glutamatergic',
                         'Esr2-IRES2-Cre-neo':            'Glutamatergic',
                         'Esr2-IRES2-Cre-neo|PhiC31-neo': 'Glutamatergic',
                         'Esr2-IRES2-Cre|PhiC31-neo':     'Glutamatergic',
                         'Rbp4-Cre_KL100':                'Glutamatergic',
                         'Scnn1a-Tg2-Cre':                'Glutamatergic',
                         'Scnn1a-Tg3-Cre':                'Glutamatergic',
                         'Rorb-IRES2-Cre':                'Glutamatergic',
                         'Cux2-CreERT2':                  'Glutamatergic',
                         'Nr5a1-Cre':                     'Glutamatergic'}

        # Discard GABAergic neurons that are not inhibitory (aspiny and sparsely spiny)
        if cell['dendrite_type'] == 'aspiny' or cell['dendrite_type'] == 'sparsely spiny':
            try:
                cell['transgenic_line'] = gabaergic[cell['transgenic_line']]
            except KeyError:
                return False

        # Discard Glutamatergic neurons that are not excitatory (spiny)
        if cell['dendrite_type'] == 'spiny':
            try:
                cell['transgenic_line'] = glutamatergic[cell['transgenic_line']]
            except KeyError:
                return False

        return True

    def generate_data(self):
        """
        :return: creates the cell ephys db and saves data into known paths with different formats.
        """
        db = {}
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
                    db[this_cell_id] = {**{'transgenic_line': cell['transgenic_line'],
                                           'dendrite_type': cell['dendrite_type'],
                                           'layer': cell['structure_layer_name']}, **ephys_feats}

        self.save_ephys_data(db, 'transcriptomic_taxonomy.csv')


if __name__ == '__main__':
    downloader = Downloader()
    downloader.generate_data()


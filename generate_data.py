from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
from tqdm import tqdm
from neuron import Neuron
import pandas as pd
import numpy as np
import os
import warnings

warnings.simplefilter("ignore")
dir_path = os.path.dirname(os.path.realpath(__file__))
path_human = dir_path + '/data/human'
path_mouse = dir_path + '/data/mouse'
path_raw_human = dir_path + '/data/raw/human'
path_raw_mouse = dir_path + '/data/raw/mouse'
sample_rate = 50000


class Downloader:
    def __init__(self, human: bool = False) -> None:
        """
        :param human: defines whether to take human cells or mouse cells for the experiment.
        """
        self.human = human
        self.ctc = CellTypesCache()
        if self.human:
            self.cells = self.ctc.get_cells(species=[CellTypesApi.HUMAN])
        else:
            self.cells = self.ctc.get_cells(species=[CellTypesApi.MOUSE])
        self.esf = EphysSweepFeatureExtractor

    def save_raw_data(self, sweep_data: dict, cell_id: str) -> None:
        """
        :param sweep_data: sweep data for cell sweep.
        :param cell_id: id of cell.
        :return: saves the raw data to a path.
        """
        file = '{}/{}.npy'.format(path_raw_mouse, cell_id)
        if self.human:
            file = '{}/{}.npy'.format(path_raw_human, cell_id)
        relevant_signal = range(*sweep_data['index_range'])
        stimulation_given = np.where(sweep_data['stimulus'][relevant_signal] > 0)[0]
        resample = int(sweep_data['sampling_rate'] / sample_rate)
        response = sweep_data['response'][relevant_signal][stimulation_given][::resample]
        np.save(file, response)

    def save_ephys_data(self, ephys_data: dict) -> None:
        """
        :param ephys_data: the cell electrophysiology data.
        :return: saves data to path in pkl and csv formats.
        """
        df = pd.DataFrame(data=ephys_data).transpose()
        df['layer'] = df['layer'].replace(['6a', '6b'], 6)
        df['layer'] = df['layer'].replace('2/3', 2)
        df['layer'] = df['layer'].astype('int')
        df['file_name'] = df.index
        if self.human:
            df.to_csv(os.path.join(path_human, 'ephys_data.csv'), index=False)
        else:
            df.to_csv(os.path.join(path_mouse, 'ephys_data.csv'), index=False)

    @staticmethod
    def get_ephys_features(sweep_data: dict) -> dict:
        """
        :param sweep_data: sweep data for cell sweep.
        :return: analysis of ephys features.
        """
        index_range = sweep_data["index_range"]
        i = sweep_data["stimulus"][0:index_range[1] + 1]  # in A
        v = sweep_data["response"][0:index_range[1] + 1]  # in V
        i *= 1e12  # to pA
        v *= 1e3  # to mV
        sampling_rate = sweep_data["sampling_rate"]  # in Hz
        t = np.arange(0, len(v)) * (1.0 / sampling_rate)
        result = {}
        sweep_ext = EphysSweepFeatureExtractor(t=t, v=v, i=i)
        sweep_ext.process_spikes()
        for key in sweep_ext.spike_feature_keys():
            try:
                result[key] = np.mean(sweep_ext.spike_feature(key))
            except TypeError:
                continue
        return result

    def generate_data(self):
        """
        :return: creates the cell ephys db and saves data into known paths with different formats.
        """
        neuron = Neuron()
        cell_db = {}
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
                self.save_raw_data(sweep_data, this_cell_id)
                if self.human:
                    cell_db[this_cell_id] = {**{'dendrite_type': cell['dendrite_type'],
                                                'layer':         cell['structure_layer_name']}, **ephys_feats}
                else:
                    if neuron.validate(cell):
                        cell_db[this_cell_id] = {**{'transgenic_line':  neuron.get_cell_transgenic_line(cell),
                                                    'neurotransmitter': neuron.get_cell_neurotransmitter(cell),
                                                    'reporter_status':  cell['reporter_status'],
                                                    'dendrite_type':    cell['dendrite_type'],
                                                    'layer':            cell['structure_layer_name']}, **ephys_feats}
        self.save_ephys_data(cell_db)


if __name__ == '__main__':
    downloader = Downloader(human=False)
    downloader.generate_data()

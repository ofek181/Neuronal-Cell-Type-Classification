# This file requires python=3.7 since allensdk does not support newer version of python

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
path_noise_human = dir_path + '/data/noise/human'
path_noise_mouse = dir_path + '/data/noise/mouse'
path_single_spike_human = dir_path + '/data/single_spike/human'
path_single_spike_mouse = dir_path + '/data/single_spike/mouse'
path_fft_human = dir_path + '/data/single_spike_fft/human'
path_fft_mouse = dir_path + '/data/single_spike_fft/mouse'
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

    def save_raw_noise_data(self, sweep_data: dict, cell_id: str) -> None:
        """
        :param sweep_data: sweep data for cell sweep.
        :param cell_id: id of cell.
        :return: saves the raw data to a path.
        """
        file_name = '{}/{}.npy'.format(path_noise_mouse, cell_id)
        if self.human:
            file_name = '{}/{}.npy'.format(path_noise_human, cell_id)
        relevant_signal = range(*sweep_data['index_range'])
        stimulation_given = np.where(sweep_data['stimulus'][relevant_signal] > 0)[0]
        resample = int(sweep_data['sampling_rate'] / sample_rate)
        response = sweep_data['response'][relevant_signal][stimulation_given][::resample]
        np.save(file_name, response)

    def save_single_spike(self, sweep_data: dict, cell_id: str, index: tuple, neuron_type: str) -> None:
        """
        :param sweep_data: sweep data for cell sweep.
        :param cell_id: id of cell.
        :param index: tuple including (start_index, end_index)
        :param neuron_type: type of neuron, spiny vs aspiny for human, t-type for mouse
        :return: saves the time series signal and FFT data to the path.
        """
        if self.human:
            if neuron_type == 'spiny':
                path_signal = path_single_spike_human + '/spiny'
                path_fft = path_fft_human + '/spiny'
            elif neuron_type == 'aspiny':
                path_signal = path_single_spike_human + '/aspiny'
                path_fft = path_fft_human + '/aspiny'
            else:
                return
        else:
            if neuron_type == 'Glutamatergic':
                path_signal = path_single_spike_mouse + '/glutamatergic'
                path_fft = path_fft_mouse + '/glutamatergic'
            elif neuron_type == 'Htr3a+|Vip-':
                path_signal = path_single_spike_mouse + '/htr3a'
                path_fft = path_fft_mouse + '/htr3a'
            elif neuron_type == 'Pvalb+':
                path_signal = path_single_spike_mouse + '/pvalb'
                path_fft = path_fft_mouse + '/pvalb'
            elif neuron_type == 'Sst+':
                path_signal = path_single_spike_mouse + '/sst'
                path_fft = path_fft_mouse + '/sst'
            elif neuron_type == 'Vip+':
                path_signal = path_single_spike_mouse + '/vip'
                path_fft = path_fft_mouse + '/vip'
            else:
                return

        signal_file_name = '{}/{}.npy'.format(path_signal, cell_id)
        response = sweep_data['response'][range(*index)]
        np.save(signal_file_name, response)

        fft_file_name = '{}/{}.npy'.format(path_fft, cell_id)
        fft_signal = np.fft.fft(response)
        np.save(fft_file_name, fft_signal)

    def save_ephys_data(self, ephys_data: dict, name: str) -> None:
        """
        :param ephys_data: the cell electrophysiology data.
        :param name: name of the file.
        :return: saves data to path in pkl and csv formats.
        """
        df = pd.DataFrame(data=ephys_data).transpose()
        df['layer'] = df['layer'].replace({'1': 'L1', '2': 'L2', '3': 'L3', '2/3': 'L2/L3',
                                           '4': 'L4', '5': 'L5', '6': 'L6', '6a': 'L6b', '6b': 'L6a'})
        df = df[df['dendrite_type'].isin(['spiny', 'aspiny'])]
        df['file_name'] = df.index
        if self.human:
            df.to_csv(os.path.join(path_human, name), index=False)
        else:
            df.to_csv(os.path.join(path_mouse, name), index=False)

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
        cell_db, single_spike_db = {}, {}
        allensdk_ephys_feats = self.ctc.get_ephys_features()
        for ind, cell in tqdm(enumerate(self.cells)):
            cell_id = cell['id']
            data_set = self.ctc.get_ephys_data(cell_id)
            sweeps = self.ctc.get_ephys_sweeps(cell_id)
            precalculated_ephys_features = next(item for item in allensdk_ephys_feats if item['specimen_id'] == cell_id)
            noise_sweep_number = [x['sweep_number'] for x in sweeps
                                  if x['stimulus_name'] in ['Noise 1', 'Noise 2']
                                  and x['num_spikes'] is not None
                                  and x['num_spikes'] > 10]
            short_square_sweep_number = [x['sweep_number'] for x in sweeps
                                         if x['stimulus_name'] in ['Short Square']
                                         and x['num_spikes'] is not None
                                         and x['num_spikes'] == 1]
            if not noise_sweep_number:
                continue

            if not short_square_sweep_number:
                continue

            try:  # Make sure ephys file is not corrupted
                sweep_data = data_set.get_sweep(noise_sweep_number[0])
                calculated_ephys_feats = self.get_ephys_features(sweep_data)
                single_spike = data_set.get_sweep(short_square_sweep_number[0])
                spike = self.get_ephys_features(single_spike)
            except:
                corrupted_filename = self.ctc.get_cache_path('', 'EPHYS_DATA', cell_id)
                os.remove(corrupted_filename)
                continue

            irrelevant_columns = ['threshold_index', 'clipped', 'threshold_t', 'peak_index', 'peak_t', 'trough_index',
                                  'trough_t', 'upstroke_index', 'upstroke_t', 'downstroke_index', 'downstroke_t',
                                  'fast_trough_index', 'fast_trough_t', 'adp_index', 'adp_t', 'adp_v', 'adp_i',
                                  'slow_trough_index', 'slow_trough_t', 'slow_trough_v', 'slow_trough_i']
            irrelevant_columns_for_single_spike = ['threshold_i', 'peak_i', 'trough_i', 'fast_trough_i']
            spike_features = pd.DataFrame(spike, index=[0])
            spike_features = spike_features.drop([x for x in irrelevant_columns
                                                  if x in spike_features.columns], axis=1, errors='ignore')
            spike_features = spike_features.drop([x for x in irrelevant_columns_for_single_spike
                                                  if x in spike_features.columns],
                                                 axis=1, errors='ignore')
            df1 = pd.DataFrame(calculated_ephys_feats, index=[0])
            df1 = df1.drop([x for x in irrelevant_columns if x in df1.columns], axis=1, errors='ignore')

            df2 = pd.DataFrame(precalculated_ephys_features, index=[0])
            irrelevant_columns = ['electrode_0_pa', 'has_burst', 'has_delay', 'has_pause', 'id', 'rheobase_sweep_id',
                                  'rheobase_sweep_number', 'specimen_id', 'thumbnail_sweep_id', 'adaptation',
                                  'avg_isi', 'slow_trough_t_long_square', 'slow_trough_t_ramp',
                                  'slow_trough_t_short_square', 'slow_trough_v_long_square', 'slow_trough_v_ramp',
                                  'slow_trough_v_short_square', 'fast_trough_t_long_square', 'fast_trough_t_ramp',
                                  'fast_trough_t_short_square', 'peak_t_long_square', 'peak_t_ramp',
                                  'peak_t_short_square', 'threshold_t_long_square', 'threshold_t_ramp',
                                  'threshold_t_short_square', 'trough_t_long square', 'trough_t_ramp',
                                  'trough_t_short_square', 'trough_t_long_square']
            df2 = df2.drop([x for x in irrelevant_columns if x in df2.columns], axis=1, errors='ignore')

            ephys_feats = pd.concat([df1, df2], axis=1).to_dict(orient='list')
            spike_features = spike_features.to_dict(orient='list')
            for key in ephys_feats:
                ephys_feats[key] = ephys_feats[key][0]
            for key in spike_features:
                spike_features[key] = spike_features[key][0]

            noise_id = '{}_{}'.format(cell_id, noise_sweep_number[0])
            square_id = '{}_{}'.format(cell_id, short_square_sweep_number[0])

            peak_index, frequency = int(spike['peak_index']), int(single_spike['sampling_rate'])
            # get index at 1ms before peak
            start = int(peak_index - frequency * 0.001)
            # get index at 2ms after peak
            end = int(peak_index + frequency * 0.002)
            # get delay from stimulus to peak
            stimulus_index = np.where(single_spike['stimulus'] > 100)[0][0]
            delay = (peak_index - stimulus_index) / frequency
            if self.human:
                cell_db[noise_id] = {**{'dendrite_type': cell['dendrite_type'],
                                        'layer': cell['structure_layer_name']}, **ephys_feats}
                single_spike_db[square_id] = {**{'dendrite_type': cell['dendrite_type'],
                                                 'layer': cell['structure_layer_name']},
                                                 'stimulus_to_peak_t': delay, **spike_features}
                self.save_single_spike(single_spike, square_id, (start, end), cell['dendrite_type'])
            else:
                if neuron.validate(cell):
                    cell_db[noise_id] = {**{'transgenic_line': neuron.get_cell_transgenic_line(cell),
                                            'neurotransmitter': neuron.get_cell_neurotransmitter(cell),
                                            'reporter_status': cell['reporter_status'],
                                            'dendrite_type': cell['dendrite_type'],
                                            'layer': cell['structure_layer_name']}, **ephys_feats}
                    single_spike_db[square_id] = {**{'transgenic_line': neuron.get_cell_transgenic_line(cell),
                                                     'neurotransmitter': neuron.get_cell_neurotransmitter(cell),
                                                     'reporter_status': cell['reporter_status'],
                                                     'dendrite_type': cell['dendrite_type'],
                                                     'layer': cell['structure_layer_name']},
                                                     'stimulus_to_peak_t': delay, **spike_features}
                    self.save_single_spike(single_spike, square_id, (start, end),
                                           neuron.get_cell_transgenic_line(cell))

            self.save_raw_noise_data(sweep_data, noise_id)

        self.save_ephys_data(cell_db, 'ephys_data.csv')
        self.save_ephys_data(single_spike_db, 'single_spike_data.csv')


if __name__ == '__main__':
    downloader = Downloader(human=True)
    downloader.generate_data()
    downloader = Downloader(human=False)
    downloader.generate_data()

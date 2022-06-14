from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings
from h5py.h5py_warnings import H5pyDeprecationWarning
warnings.filterwarnings('ignore', category=H5pyDeprecationWarning)


class Downloader:
    def __init__(self, human: bool = False, data_path: str = None, out_dir: str = None) -> None:
        self.ctc = CellTypesCache()
        if human:
            self.cells = self.ctc.get_cells(species=[CellTypesApi.HUMAN])
        else:
            self.cells = self.ctc.get_cells(species=[CellTypesApi.MOUSE])
        self.esf = EphysSweepFeatureExtractor
        self.data_path = data_path
        self.sample_rate = 50000
        self.out_dir = out_dir

    def download(self):
        pass

    def create_data(self):
        cell_db = {}
        for ind, cell in enumerate(self.cells):
            cell_id = cell['id']
            data_set = self.ctc.get_ephys_data(cell_id)
            sweeps = self.ctc.get_ephys_sweeps(cell_id)
            noise_sweep_number = [x['sweep_number'] for x in sweeps
                                  if x['stimulus_name'] in ['Noise 1', 'Noise 2']
                                  and x['num_spikes'] is not None
                                  and x['num_spikes'] > 15]
            if not noise_sweep_number:
                continue

            try:  # Make sure ephys file is not corrupted
                sweep_data = data_set.get_sweep(noise_sweep_number[0])
            except:
                corrupted_filename = self.ctc.get_cache_path('', 'EPHYS_DATA', cell_id)
                os.remove(corrupted_filename)
                data_set = self.ctc.get_ephys_data(cell_id)

            for sweep_num in [noise_sweep_number[0]]:
                print('Proccesing cell: {} sweep: {}. Cell {}/{}'.format(cell_id, sweep_num,
                                                                         ind + 1, len(self.cells)))
                this_cell_id = '{}_{}'.format(cell_id, sweep_num)
                sweep_data = data_set.get_sweep(sweep_num)
                ephys_feats = self.get_ephys_features(sweep_data)
                raw_data_file = '{}/{}_data.npy'.format(self.out_dir, this_cell_id)
                raw_data_stim = '{}{}_stim.npy'.format(self.data_path, this_cell_id)
                relevant_signal = range(*sweep_data['index_range'])
                stimulation_given = np.where(sweep_data['stimulus'][relevant_signal] > 0)[0]
                resample = int(sweep_data['sampling_rate'] / self.sample_rate)
                response = sweep_data['response'][relevant_signal][stimulation_given][::resample]
                np.save(raw_data_file, response)  # .astype('float16'))
                np.save(raw_data_file, response.astype('float16'))
                np.save(raw_data_stim, stimulation_given)
                cell_db[this_cell_id] = {**{'layer': cell['structure_layer_name'],
                                            'dendrite_type': cell['dendrite_type'],
                                            'structure_area_abbrev': cell['structure_area_abbrev'],
                                            'sampling_rate': sweep_data['sampling_rate']}, **ephys_feats}
                # 'stimulation_given': raw_data_stim}

        df = pd.DataFrame(data=cell_db).transpose()
        df['sampling_rate'] = df['sampling_rate'].astype('float')
        df['layer'] = df['layer'].replace(['6a', '6b'], 6)
        df['layer'] = df['layer'].replace('2/3', 2)
        df['layer'] = df['layer'].astype('int')
        df = df[df['dendrite_type'].isin(['spiny', 'aspiny'])]
        df['file_name'] = df.index
        with open(self.data_path + '/cells/db.p', 'wb') as f:
            pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def get_ephys_features(sweep_data):
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
                result['mean_' + key] = np.mean(sweep_ext.spike_feature(key))
            except TypeError:
                continue
        return result


if __name__ == '__main__':
    downloader = Downloader(data_path='cell_types/', out_dir='cell_types/')
    downloader.create_data()
    """
    plt.subplot(2,1,1)
    plt.plot(np.linspace(0,9,len(relevant_signal)), sweep_data['stimulus'][relevant_signal])
    plt.xlabel('Time [sec]')
    plt.ylabel('mV')
    plt.title('Stimulation')

    plt.subplot(2,1,2)
    plt.plot(np.linspace(0,9,len(relevant_signal)), sweep_data['response'][relevant_signal])
    plt.xlabel('Time [sec]')
    plt.ylabel('mV')
    plt.title('Recording')
    plt.subplots_adjust(hspace=0.5)

    """



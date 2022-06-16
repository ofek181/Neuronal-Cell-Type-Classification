from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings
from gramian_angular_field import activity_to_image_gaf
from imageio import imwrite

warnings.filterwarnings('ignore')

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path + '\\cell_types'
images_path = data_path + '\\images'

# TODO add docstrings
# TODO delete non used functions and make code more readable
# TODO fix pathing of data creation

class Downloader:
    def __init__(self, human: bool = False, out_dir: str = None) -> None:
        self.ctc = CellTypesCache()
        if human:
            self.cells = self.ctc.get_cells(species=[CellTypesApi.HUMAN])
        else:
            self.cells = self.ctc.get_cells(species=[CellTypesApi.MOUSE])
        self.esf = EphysSweepFeatureExtractor
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
                ephys_feats = self.get_ephys_features(sweep_data)
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
                raw_data_file = '{}/{}.npy'.format(data_path, this_cell_id)
                relevant_signal = range(*sweep_data['index_range'])
                stimulation_given = np.where(sweep_data['stimulus'][relevant_signal] > 0)[0]
                resample = int(sweep_data['sampling_rate'] / self.sample_rate)
                response = sweep_data['response'][relevant_signal][stimulation_given][::resample]
                response_img = activity_to_image_gaf(response)
                image_save_location = '{}{}.png'.format(images_path, this_cell_id)
                imwrite(image_save_location, response_img)
                np.save(raw_data_file, response)
                cell_db[this_cell_id] = {**{'layer': cell['structure_layer_name'],
                                            'dendrite_type': cell['dendrite_type'],
                                            'structure_area_abbrev': cell['structure_area_abbrev'],
                                            'sampling_rate': sweep_data['sampling_rate']}, **ephys_feats}
                # print(cell_db[this_cell_id])

        df = pd.DataFrame(data=cell_db).transpose()
        df['sampling_rate'] = df['sampling_rate'].astype('float')
        df['layer'] = df['layer'].replace(['6a', '6b'], 6)
        df['layer'] = df['layer'].replace('2/3', 2)
        df['layer'] = df['layer'].astype('int')
        df = df[df['dendrite_type'].isin(['spiny', 'aspiny'])]
        df['file_name'] = df.index
        with open(data_path + '\\ephys_data.pkl', 'wb') as f:
            pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def save_pkl_as_csv():
        with open(data_path + "\\ephys_data.pkl", "rb") as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(data_path, 'ephys_data.csv'))

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
    downloader = Downloader(human=True, out_dir='cell_types/')
    downloader.create_data()
    # downloader.save_pkl_as_csv()
    # downloader.create_ephys_csv_data()
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

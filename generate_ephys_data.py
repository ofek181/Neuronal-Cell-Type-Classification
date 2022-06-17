from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
import pandas as pd
import numpy as np
import os
import warnings
from time_series_to_image import activity_to_image_gaf
from imageio import imwrite

warnings.filterwarnings('ignore')

dir_path = os.path.dirname(os.path.realpath(__file__))

dataframe_path = dir_path + '\\data\\dataframe'
raw_data_path = dir_path + '\\data\\raw_data'
gaf_path = dir_path + '\\data\\images\\gaf'

sample_rate = 50000


class Downloader:
    def __init__(self, human: bool = False) -> None:
        """
        :param human: defines whether to take human cells or mouse cells for the experiment.
        """
        self.ctc = CellTypesCache()
        if human:
            self.cells = self.ctc.get_cells(species=[CellTypesApi.HUMAN])
        else:
            self.cells = self.ctc.get_cells(species=[CellTypesApi.MOUSE])
        self.esf = EphysSweepFeatureExtractor

    @staticmethod
    def _save_gaf_image(response: np.ndarray, cell_id: str) -> None:
        """
        :param response: response for the sweep data of the cell.
        :param cell_id: id of cell.
        :return: saves a Gramian Angular Field image to path.
        """
        response_gaf = activity_to_image_gaf(response)
        image_name = '{}_gaf.png'.format(cell_id)
        imwrite(os.path.join(gaf_path, image_name), response_gaf)

    @staticmethod
    def save_gaf_and_raw_data(sweep_data: dict, cell_id: str) -> None:
        """
        :param sweep_data: sweep data for cell sweep.
        :param cell_id: id of cell.
        :return: saves a Gramian Angular Field image to path and raw data to a different path.
        """
        raw_data_file = '{}/{}.npy'.format(raw_data_path, cell_id)
        relevant_signal = range(*sweep_data['index_range'])
        stimulation_given = np.where(sweep_data['stimulus'][relevant_signal] > 0)[0]
        resample = int(sweep_data['sampling_rate'] / sample_rate)
        response = sweep_data['response'][relevant_signal][stimulation_given][::resample]
        np.save(raw_data_file, response)
        Downloader._save_gaf_image(response, cell_id)

    @staticmethod
    def save_ephys_data(ephys_data: dict) -> None:
        """
        :param ephys_data: the cell electrophysiology data.
        :return: saves data to path in pkl and csv formats.
        """
        df = pd.DataFrame(data=ephys_data).transpose()
        df['sampling_rate'] = df['sampling_rate'].astype('float')
        df['layer'] = df['layer'].replace(['6a', '6b'], 6)
        df['layer'] = df['layer'].replace('2/3', 2)
        df['layer'] = df['layer'].astype('int')
        df = df[df['dendrite_type'].isin(['spiny', 'aspiny'])]
        df['file_name'] = df.index

        # with open(dataframe_path + '\\ephys_data.pkl', 'wb') as f:
        #     pickle.dump(df, f)

        df.to_csv(os.path.join(dataframe_path, 'extracted_mean_ephys_data.csv'))

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
                result['mean_' + key] = np.mean(sweep_ext.spike_feature(key))
            except TypeError:
                continue

        return result

    def create_ephys_dataframe(self) -> None:
        """
        :return: saves a ephys feature dataframe with the allensdk
        """
        data = self.ctc.get_ephys_features(dataframe=True)
        data.to_csv(os.path.join(dataframe_path, 'ephys_features.csv'))

    def generate_data(self):
        """
        :return: creates the cell ephys db and saves data into known paths with different formats.
        """
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
                print('Processing cell: {} sweep: {}. Cell {}/{}'.format(cell_id, sweep_num,
                                                                         ind + 1, len(self.cells)))
                this_cell_id = '{}_{}'.format(cell_id, sweep_num)
                sweep_data = data_set.get_sweep(sweep_num)
                ephys_feats = self.get_ephys_features(sweep_data)
                self.save_gaf_and_raw_data(sweep_data, this_cell_id)
                cell_db[this_cell_id] = {**{'layer': cell['structure_layer_name'],
                                            'dendrite_type': cell['dendrite_type'],
                                            'structure_area_abbrev': cell['structure_area_abbrev'],
                                            'sampling_rate': sweep_data['sampling_rate']}, **ephys_feats}
        self.save_ephys_data(cell_db)


if __name__ == '__main__':
    downloader = Downloader(human=True)
    downloader.generate_data()
    downloader.create_ephys_dataframe()

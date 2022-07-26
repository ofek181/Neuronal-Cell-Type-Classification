from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
import pandas as pd
import numpy as np
import os
from consts import GAF_IMAGE_SIZE
from time_series_to_image import activity_to_gaf_rgb, activity_to_gaf
from imageio import imwrite
from tqdm import tqdm
import warnings

# TODO add grayscale gaf maybe it will help

warnings.simplefilter("ignore")
dir_path = os.path.dirname(os.path.realpath(__file__))
dataframe_path = dir_path + '/data/dataframe'
dataframe_path_human = dataframe_path + '/human'
dataframe_path_mouse = dataframe_path + '/mouse'
raw_data_path = dir_path + '/data/raw_data'
raw_data_path_human = raw_data_path + '/human'
raw_data_path_mouse = raw_data_path + '/mouse'
gaf_path = dir_path + '/data/images/gaf_rgb'
gaf_gray_path = dir_path + '/data/images/gaf_gray'
gaf_npy_path = gaf_path + '/npy'
gaf_npy_path_human = gaf_npy_path + '/human'
gaf_npy_path_mouse = gaf_npy_path + '/mouse'
gaf_png_path = gaf_path + '/png'
gaf_png_path_human = gaf_png_path + '/human'
gaf_png_path_mouse = gaf_png_path + '/mouse'
gaf_npy_path_gray = gaf_gray_path + '/npy'
gaf_npy_path_human_gray = gaf_npy_path_gray + '/human'
gaf_npy_path_mouse_gray = gaf_npy_path_gray + '/mouse'
gaf_png_path_gray = gaf_gray_path + '/png'
gaf_png_path_human_gray = gaf_png_path_gray + '/human'
gaf_png_path_mouse_gray = gaf_png_path_gray + '/mouse'

sample_rate = 50000
idx = 0


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
        self.human = human
        self.esf = EphysSweepFeatureExtractor
        rgb = 3
        self.images_rgb = np.zeros((len(self.cells), GAF_IMAGE_SIZE, GAF_IMAGE_SIZE, rgb))
        self.images_gray = np.zeros((len(self.cells), GAF_IMAGE_SIZE, GAF_IMAGE_SIZE, rgb))
        self.labels = np.zeros(len(self.cells))

    def _save_gaf_image(self, response: np.ndarray, cell_id: str) -> None:
        """
        :param response: response for the sweep data of the cell.
        :param cell_id: id of cell.
        :return: saves a Gramian Angular Field image to path.
        """
        global idx
        response_gaf_rgb = activity_to_gaf_rgb(response)
        response_gaf_gray = activity_to_gaf(response)
        image_name = '{}_gaf.png'.format(cell_id)
        scaled_image_rgb = 255 * (response_gaf_rgb + 1) / 2
        scaled_image_gray = 255 * (response_gaf_gray + 1) / 2
        img_rgb = scaled_image_rgb.astype(np.uint8)
        img_gray = scaled_image_gray.astype(np.uint8)
        self.images_rgb[idx, :, :, :] = img_rgb
        self.images_gray[idx, :, :, :] = img_gray
        if self.cells[idx]['dendrite_type'] == 'aspiny':
            self.labels[idx] = 0
            if self.human:
                imwrite(os.path.join(gaf_png_path_human + '/aspiny', image_name), img_rgb)
                imwrite(os.path.join(gaf_png_path_human_gray + '/aspiny', image_name), img_gray)
            else:
                imwrite(os.path.join(gaf_png_path_mouse + '/aspiny', image_name), img_rgb)
                imwrite(os.path.join(gaf_png_path_mouse_gray + '/aspiny', image_name), img_gray)
        if self.cells[idx]['dendrite_type'] == 'spiny':
            self.labels[idx] = 1
            if self.human:
                imwrite(os.path.join(gaf_png_path_human + '/spiny', image_name), img_rgb)
                imwrite(os.path.join(gaf_png_path_human_gray + '/spiny', image_name), img_gray)
            else:
                imwrite(os.path.join(gaf_png_path_mouse + '/spiny', image_name), img_rgb)
                imwrite(os.path.join(gaf_png_path_mouse_gray + '/spiny', image_name), img_gray)
        idx += 1

    def save_gaf_and_raw_data(self, sweep_data: dict, cell_id: str) -> None:
        """
        :param sweep_data: sweep data for cell sweep.
        :param cell_id: id of cell.
        :return: saves a Gramian Angular Field image to path and raw data to a different path.
        """
        if self.human:
            raw_data_file = '{}/{}.npy'.format(raw_data_path_human, cell_id)
        else:
            raw_data_file = '{}/{}.npy'.format(raw_data_path_mouse, cell_id)
        relevant_signal = range(*sweep_data['index_range'])
        stimulation_given = np.where(sweep_data['stimulus'][relevant_signal] > 0)[0]
        resample = int(sweep_data['sampling_rate'] / sample_rate)
        response = sweep_data['response'][relevant_signal][stimulation_given][::resample]
        np.save(raw_data_file, response)
        self._save_gaf_image(response, cell_id)

    def save_ephys_data(self, ephys_data: dict) -> None:
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
        if self.human:
            df.to_csv(os.path.join(dataframe_path_human, 'extracted_mean_ephys_data.csv'), index=False)
        else:
            df.to_csv(os.path.join(dataframe_path_mouse, 'extracted_mean_ephys_data.csv'), index=False)

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
        data.to_csv(os.path.join(dataframe_path, 'ephys_features.csv'), index=False)

    def generate_data(self):
        """
        :return: creates the cell ephys db and saves data into known paths with different formats.
        """
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
                self.save_gaf_and_raw_data(sweep_data, this_cell_id)
                cell_db[this_cell_id] = {**{'layer': cell['structure_layer_name'],
                                            'dendrite_type': cell['dendrite_type'],
                                            'structure_area_abbrev': cell['structure_area_abbrev'],
                                            'sampling_rate': sweep_data['sampling_rate']}, **ephys_feats}
        self.save_ephys_data(cell_db)
        if self.human:
            imgs_rgb = '{}/images.npy'.format(gaf_npy_path_human)
            imgs_gray = '{}/images.npy'.format(gaf_npy_path_human_gray)
            lbls_rgb = '{}/labels.npy'.format(gaf_npy_path_human)
            lbls_gray = '{}/labels.npy'.format(gaf_npy_path_human_gray)
        else:
            imgs_rgb = '{}/images.npy'.format(gaf_npy_path_mouse)
            imgs_gray = '{}/images.npy'.format(gaf_npy_path_mouse_gray)
            lbls_rgb = '{}/labels.npy'.format(gaf_npy_path_mouse)
            lbls_gray = '{}/labels.npy'.format(gaf_npy_path_mouse_gray)
        np.save(imgs_rgb, self.images_rgb)
        np.save(imgs_gray, self.images_gray)
        np.save(lbls_rgb, self.labels)
        np.save(lbls_gray, self.labels)


if __name__ == '__main__':
    downloader = Downloader(human=False)
    downloader.generate_data()
    downloader.create_ephys_dataframe()

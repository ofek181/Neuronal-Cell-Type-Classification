from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        self.ctc = CellTypesCache()
        self.cells = self.ctc.get_cells(species=[CellTypesApi.MOUSE])
        self.sweep_data = self._get_stimulation_and_response()

    def plot_stimulation_and_response(self):
        relevant_signal = range(*self.sweep_data['index_range'])
        stimulation = self.sweep_data['stimulus'][relevant_signal]  # in A
        stimulation *= 1e12  # to pA
        response = self.sweep_data['response'][relevant_signal]  # in V
        response *= 1e3  # to mV

        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(0, 9, len(relevant_signal)), stimulation, 'b-', linewidth=0.5, alpha=0.7)
        plt.xlabel('Seconds')
        plt.ylabel('pA')
        plt.title('Stimulation')

        plt.subplot(2, 1, 2)
        plt.plot(np.linspace(0, 9, len(relevant_signal)), response, 'r-', linewidth=0.1, alpha=0.7)
        plt.xlabel('Seconds')
        plt.ylabel('mV')
        plt.ylim([-80, 50])
        plt.title('Response')

        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def _get_stimulation_and_response(self):
        for ind, cell in enumerate(self.cells):
            cell_id = cell['id']
            data_set = self.ctc.get_ephys_data(cell_id)
            sweeps = self.ctc.get_ephys_sweeps(cell_id)
            noise_sweep_number = [x['sweep_number'] for x in sweeps
                                  if x['stimulus_name'] in ['Noise 1', 'Noise 2']
                                  and x['num_spikes'] is not None
                                  and x['num_spikes'] > 10]
            if not noise_sweep_number:
                continue

            try:
                sweep_data = data_set.get_sweep(noise_sweep_number[0])
                return sweep_data
            except:
                continue


def main():
    plotter = Plotter()
    plotter.plot_stimulation_and_response()


if __name__ == '__main__':
    main()


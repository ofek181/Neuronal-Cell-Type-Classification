from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
excitatory_data = pd.read_csv(dir_path + '/data/dataframe/mouse/excitatory_transgenic_data.csv')
inhibitory_data = pd.read_csv(dir_path + '/data/dataframe/mouse/inhibitory_transgenic_data.csv')
all_data = pd.read_csv(dir_path + '/data/dataframe/mouse/transcriptomic_taxonomy.csv')
mouse_data = pd.read_csv(dir_path + '/data/dataframe/mouse/extracted_mean_ephys_data.csv')
human_data = pd.read_csv(dir_path + '/data/dataframe/human/extracted_mean_ephys_data.csv')


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

        fig, axes = plt.subplots(2, 1)
        axes[0].plot(np.linspace(0, 9, len(relevant_signal)), stimulation, 'b-', linewidth=0.5, alpha=0.7)
        axes[0].set_xlabel('Seconds')
        axes[0].set_ylabel('pA')
        axes[0].set_title('Stimulation')

        axes[1].plot(np.linspace(0, 9, len(relevant_signal)), response, 'r-', linewidth=0.1, alpha=0.7)
        axes[1].set_xlabel('Seconds')
        axes[1].set_ylabel('mV')
        axes[1].set_ylim([-80, 50])
        axes[1].set_title('Response')

        plt.subplots_adjust(hspace=0.5)
        fig.suptitle('Electrophysiological activity of a neuron')
        plt.draw()

    @staticmethod
    def plot_excitatory_inhibitory_distribution():
        fig, axes = plt.subplots(1, 2, sharey='all')
        fig.suptitle('Distribution of inhibitory/excitatory cells')

        axes[0].set_title('Mouse data')
        ax = sns.countplot(ax=axes[0], x="dendrite_type", data=mouse_data, palette="Set2")
        for container in ax.containers:
            ax.bar_label(container)

        axes[1].set_title('Human data')
        ax = sns.countplot(ax=axes[1], x="dendrite_type", data=human_data, palette="Set2")
        for container in ax.containers:
            ax.bar_label(container)

        plt.tight_layout()
        plt.draw()

    @staticmethod
    def plot_transcriptomatic_distribution():
        plt.figure()
        plt.title('Distribution of marker genes in excitatory cells')
        ax = sns.countplot(x="transgenic_line", data=excitatory_data, palette="Set2",
                           order=excitatory_data['transgenic_line'].value_counts().index)
        for container in ax.containers:
            ax.bar_label(container)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.draw()

        plt.figure()
        plt.title('Distribution of marker genes in inhibitory cells')
        ax = sns.countplot(x="transgenic_line", data=inhibitory_data, palette="Set2",
                           order=inhibitory_data['transgenic_line'].value_counts().index)
        for container in ax.containers:
            ax.bar_label(container)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.draw()

    @staticmethod
    def plot_merged_gene_distribution():
        plt.figure()
        plt.title('Distribution of t-types (merged)')
        ax = sns.countplot(x="transgenic_line", data=all_data, palette="Set2",
                           order=all_data['transgenic_line'].value_counts().index)
        for container in ax.containers:
            ax.bar_label(container)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.draw()

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
    plotter.plot_excitatory_inhibitory_distribution()
    plotter.plot_transcriptomatic_distribution()
    plotter.plot_merged_gene_distribution()
    plt.show()


if __name__ == '__main__':
    main()


class Neuron:
    def __init__(self) -> None:
        """
        Defines neuron type based on Cre lines and Neurotransmitter type:
                Cre lines that have similar characteristics are kept together
                Zeng, Hongkui, and Joshua R. Sanes.
                "Neuronal cell-type classification: challenges, opportunities and the path forward."
                Nature Reviews Neuroscience 18.9 (2017): 530-546.
                Tasic, Bosiljka, et al.
                "Adult mouse cortical cell taxonomy revealed by single cell transcriptomics."
                Nature neuroscience 19.2 (2016): 335-346.
        """
        transmitter = {'Pvalb-IRES-Cre':                     'GABAergic',
                       'Slc32a1-T2A-FlpO|Vipr2-IRES2-Cre':   'GABAergic',
                       'Nkx2-1-CreERT2':                     'GABAergic',
                       'Ndnf-IRES2-dgCre':                   'GABAergic',
                       'Gad2-IRES-Cre':                      'GABAergic',
                       'Htr3a-Cre_NO152':                    'GABAergic',
                       'Sst-IRES-Cre':                       'GABAergic',
                       'Nos1-CreERT2':                       'GABAergic',
                       'Nos1-CreERT2|Sst-IRES-FlpO':         'GABAergic',
                       'Oxtr-T2A-Cre':                       'GABAergic',
                       'Chrna2-Cre_OE25':                    'GABAergic',
                       'Chat-IRES-Cre-neo':                  'GABAergic',
                       'Vip-IRES-Cre':                       'GABAergic',
                       'Ctgf-T2A-dgCre':                     'Glutamatergic',
                       'Tlx3-Cre_PL56':                      'Glutamatergic',
                       'Sim1-Cre_KJ18':                      'Glutamatergic',
                       'Glt25d2-Cre_NF107':                  'Glutamatergic',
                       'Ntsr1-Cre_GN220':                    'Glutamatergic',
                       'Esr2-IRES2-Cre':                     'Glutamatergic',
                       'Esr2-IRES2-Cre-neo':                 'Glutamatergic',
                       'Esr2-IRES2-Cre-neo|PhiC31-neo':      'Glutamatergic',
                       'Esr2-IRES2-Cre|PhiC31-neo':          'Glutamatergic',
                       'Rbp4-Cre_KL100':                     'Glutamatergic',
                       'Scnn1a-Tg2-Cre':                     'Glutamatergic',
                       'Scnn1a-Tg3-Cre':                     'Glutamatergic',
                       'Rorb-IRES2-Cre':                     'Glutamatergic',
                       'Cux2-CreERT2':                       'Glutamatergic',
                       'Nr5a1-Cre':                          'Glutamatergic'}

        metaclasses = {'Pvalb-IRES-Cre':                     'Pvalb+',
                       'Slc32a1-T2A-FlpO|Vipr2-IRES2-Cre':   'Pvalb+',
                       'Nkx2-1-CreERT2':                     'Pvalb+',
                       'Ndnf-IRES2-dgCre':                   'Htr3a+|Vip-',
                       'Gad2-IRES-Cre':                      'Htr3a+|Vip-',
                       'Htr3a-Cre_NO152':                    'Htr3a+|Vip-',
                       'Sst-IRES-Cre':                       'Sst+',
                       'Nos1-CreERT2':                       'Sst+',
                       'Nos1-CreERT2|Sst-IRES-FlpO':         'Sst+',
                       'Oxtr-T2A-Cre':                       'Sst+',
                       'Chrna2-Cre_OE25':                    'Sst+',
                       'Chat-IRES-Cre-neo':                  'Vip+',
                       'Vip-IRES-Cre':                       'Vip+',
                       'Ctgf-T2A-dgCre':                     'Glutamatergic',
                       'Tlx3-Cre_PL56':                      'Glutamatergic',
                       'Sim1-Cre_KJ18':                      'Glutamatergic',
                       'Glt25d2-Cre_NF107':                  'Glutamatergic',
                       'Ntsr1-Cre_GN220':                    'Glutamatergic',
                       'Esr2-IRES2-Cre':                     'Glutamatergic',
                       'Esr2-IRES2-Cre-neo':                 'Glutamatergic',
                       'Esr2-IRES2-Cre-neo|PhiC31-neo':      'Glutamatergic',
                       'Esr2-IRES2-Cre|PhiC31-neo':          'Glutamatergic',
                       'Rbp4-Cre_KL100':                     'Glutamatergic',
                       'Scnn1a-Tg2-Cre':                     'Glutamatergic',
                       'Scnn1a-Tg3-Cre':                     'Glutamatergic',
                       'Rorb-IRES2-Cre':                     'Glutamatergic',
                       'Cux2-CreERT2':                       'Glutamatergic',
                       'Nr5a1-Cre':                          'Glutamatergic'}

        self.neurotransmitter = transmitter
        self.transgenic_line = metaclasses

    def get_cell_neurotransmitter(self, cell: dict) -> bool:
        """
        :param cell: metadata about the cell
        :return: neurotransmitter type
        """
        try:
            return self.neurotransmitter[cell['transgenic_line']]
        except KeyError:
            return False

    def get_cell_transgenic_line(self, cell: dict) -> bool:
        """
        :param cell: metadata about the cell
        :return: transgenic line metaclass type
        """
        try:
            return self.transgenic_line[cell['transgenic_line']]
        except KeyError:
            return False

    def validate(self, cell: dict) -> bool:
        """
        :param cell: metadata about the cell
        :return: True whether to continue using the cell based on the described tests, else False
        """
        # Discard GABAergic neurons that are not inhibitory (aspiny)
        if cell['dendrite_type'] == 'aspiny':
            try:
                neurotransmitter = self.neurotransmitter[cell['transgenic_line']]
                if neurotransmitter != 'GABAergic':
                    return False
            except KeyError:
                return False

        # Discard Glutamatergic neurons that are not excitatory (spiny)
        if cell['dendrite_type'] == 'spiny':
            try:
                neurotransmitter = self.neurotransmitter[cell['transgenic_line']]
                if neurotransmitter != 'Glutamatergic':
                    return False
            except KeyError:
                return False

        return True

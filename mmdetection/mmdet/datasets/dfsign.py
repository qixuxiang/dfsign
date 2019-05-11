from .xml_style import XMLDataset


class DFSignDataset(XMLDataset):

    CLASSES = ('1', '2', '3', '4', '5', '6', '7',
               '8', '9', '10', '11', '12', '13',
               '14', '15', '16', '17', '18', '19',
               '20', '21')

    def __init__(self, **kwargs):
        super(DFSignDataset, self).__init__(**kwargs)

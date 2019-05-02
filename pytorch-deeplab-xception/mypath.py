import os
user_home = os.path.expanduser('~')

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return os.path.join(user_home, 'data/VOCdevkit/VOC2012')  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'dfsign':
            return os.path.join(user_home, 'data/dfsign/dfsign_region_voc')  # folder that contains VOCdevkit/.    
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

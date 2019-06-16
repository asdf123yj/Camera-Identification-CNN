class ImageCollector:
    def __init__(self):
        self.imgs = dict()

    def __getitem__(self, item):
        if item not in self.imgs.keys():
            self.imgs[item] = []
        return self.imgs[item]


class FeatureCollector:
    def __init__(self):
        self.fingerprints = dict()
        self.moments = dict()
        self.cross_correlations = dict()
        self.linear_correlations = dict()
        self.block_covariances_1 = dict()
        self.block_covariances_2 = dict()

from networks.encoder import DisjointEncoder

class Backbones:
    def __init__(self, args) -> None:
        self.args = args
        self.backbones = {'fgvc': self.fgvc}
    
    def fgvc(self):
        args = self.args
        return DisjointEncoder(num_classes=args.n_classes, num_local=args.n_local, crop_mode=args.crop_mode)
    
    def get(self, task):
        return self.backbones[task]()

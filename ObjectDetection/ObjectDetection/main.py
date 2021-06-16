from suportFunctions import timer, system_init
from config import *
from ObjectDetection.training import train
from ObjectDetection.vallidation import val


def main(cfg=None):
    # Set up of algorithem

    system_init(cfg)

    #cfg.DEVICE = "cpu"

    if 'train' in cfg.STATES:
        train(cfg)

    if 'val' in cfg.STATES:
        val(cfg)

    if 'wandb' in cfg.STATES:
        wandb.finish()


if __name__ == '__main__':
    print(os.chdir('..'))

    cfg = config(clasificationb_algorithem=8,
                 clustering_algorithem=1)

    main(cfg)


import M2A.cfgs.config as config
import argparse, yaml
import random
from easydict import EasyDict as edict


def parse_args():
    parser = argparse.ArgumentParser(description='BNN Args')

    parser.add_argument('--run', dest='run_mode',
                        choices=['train', 'val', 'test', 'g_test', 'case'],
                        help='{train, val, test}',
                        type=str, required=True)

    parser.add_argument('--dataset', dest='dataset',
                        choices=['imdb', 'yelp_13', 'yelp_14', 'mtl'],
                        default='imdb', type=str)

    parser.add_argument('--gpu', dest='gpu',
                        help="gpu select, eg.'0, 1, 2'",
                        type=str,
                        default="0, 1")

    parser.add_argument('--no_cuda',
                        action='store_true',
                        default=False,
                        )

    parser.add_argument('--seed', dest='seed',
                        help='fix random seed',
                        type=int,
                        default=random.randint(0, 99999999))

    parser.add_argument('--version', dest='version',
                        help='version control',
                        type=str,
                        default="default")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    __C = config.__C

    args = parse_args()
    try:
        with open("cfgs/fga_model.yml", 'r') as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        args_dict = edict({**yaml_dict, **vars(args)})
    except:
        args_dict = edict({**vars(args)})
    config.add_edit(args_dict, __C)
    config.proc(__C)

    print('Hyper Parameters:')
    config.config_print(__C)

    from M2A.trainer.trainer import BayesianTrainer
    trainer = BayesianTrainer(__C)
    trainer.run(__C.run_mode)

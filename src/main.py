# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SMORE', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--mg', action='store_true', default=False)

    config_dict = {
        'gpu_id': 0,
    }

    args, unknown = parser.parse_known_args()

    # Parse extra key=value style arguments, e.g. seed=999 freq_band_gating=True
    def _parse_value(v):
        if v.lower() == 'true':
            return True
        if v.lower() == 'false':
            return False
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    for token in unknown:
        if '=' in token:
            key, val = token.split('=', 1)
            config_dict[key] = _parse_value(val)

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True, mg=args.mg)



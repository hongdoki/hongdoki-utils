import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tinydb import TinyDB, Query, where
import pandas as pd
import argparse


def contain_fn(ptn):
    return lambda c: ptn in c


def show_diff_hparams(hps1, hps2, key):
    df_diff = pd.DataFrame(columns=['hyper-param', hps1[key], hps2[key]])
    for k in list(set(hps1.keys()).union(hps2.keys())):
        if k == key:
            continue
        elif k not in hps1:
            df_diff.loc[df_diff.shape[0]] = [k, None, hps2[k]]
        elif k not in hps2:
            df_diff.loc[df_diff.shape[0]] = [k, hps1[k], None]
        elif hps1[k] != hps2[k]:
            df_diff.loc[df_diff.shape[0]] = [k, hps1[k], hps2[k]]

    return df_diff


def main(flags):
    # load expids
    expids = [s for s in os.listdir(flags.exps_dir) if '201' in s]
    print expids

    # import scalar from tf event
    expid_scalar_dict = {}
    for i, expid in enumerate(expids):
        event_acc = EventAccumulator(os.path.join(flags.exps_dir, expid, 'eval'))
        event_acc.Reload()
        expid_scalar_dict[expid] = [event_acc.Scalars(scalar)[0].value for scalar in flags.scalars]
        print '\rreading event files... %d/%d done.' % (i + 1, len(expids)),

    # load hyper parameters
    db = TinyDB(flags.hp_path)
    hp = db.table(flags.hp_tbname)

    hpar_default = hp.search(where(flags.expid_name).test(contain_fn(expids[0])))[0]
    print 'defalut expid: %s' % hpar_default.pop(flags.expid_name)

    for name in flags.target_hps:
        hpar_default.pop(name)
    # hpar_default.pop('visit_weight')

    results = pd.DataFrame(columns=[name for name in flags.target_hps] + flags.scalars)

    # check difference except target hpar
    for i, expid in enumerate(expid_scalar_dict.keys()):
        hpar = hp.search(where(flags.expid_name).test(contain_fn(expid)))[0]
        results.loc[i] = [hpar.pop(name) for name in flags.target_hps] + expid_scalar_dict[expid]
        hpar.pop(flags.expid_name)
        if not hpar_default == hpar:
            print expid
            print show_diff_hparams(hpar, hpar_default, 'note')

    # sort
    for name in flags.target_hps:
        results = results.sort_values(by=name)

    # save result
    if not os.path.exists(flags.out_dir):
        os.makedirs(flags.out_dir)

    results.to_csv(os.path.join(flags.out_dir, '%s.csv' % (hpar_default['note'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--exps_dir',
        type=str,
        default='./tmp',
        help='Directory which has sub-directory of target experiments'
    )

    parser.add_argument(
        '--hp_path',
        type=str,
        default='./result.json',
        help='Path for json file(TinyDB) that has hyper parameters'
    )
    parser.add_argument(
        '--hp_tbname',
        type=str,
        default='lba-hyper-params',
        help='Name of Table that has hyper parameters'
    )

    parser.add_argument(
        '--expid_name',
        type=str,
        default='logdir',
        help='Name of expreiment ID used in hyper parameters DB'
    )

    parser.add_argument(
        '--scalars',
        '--scalars',
        nargs='+',
        default=['AUC_validation','AUC_test'],
        help='list of names for scalars in TF event file to be used as target value(y)'
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        default='./output',
        help='Path for a output csv file'
    )

    parser.add_argument(
        '--target_hps',
        '--target_hps',
        nargs='+',
        default=['emb_size'],
        help='list of names for target hyper-parmeters'
    )

    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)

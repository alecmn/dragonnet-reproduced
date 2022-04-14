from semi_parametric_estimation.ate import *
from numpy import load
import pandas as pd
import numpy as np
import copy
import glob
import os


def load_truth(file_path='../../dat/LIBDD/scaling/params.csv',
               ufid='00ea30e866f141d9880d5824a361a76a'):
    """
    loading ground truth data
    """
    df = pd.read_csv(file_path)
    res = df[df['ufid'] == str(ufid)]
    truth = np.squeeze(res.effect_size.values)

    return truth


def load_data(ufid='00ea30e866f141d9880d5824a361a76a', model='baseline', train_test='test', replication=0,
              file_path=''):
    """
    loading train test experiment results
    """

    data = load(file_path + '{}/{}/{}_replication_{}.npz'.format(ufid, model, str(replication), train_test))
    q_t0 = data['q_t0'].reshape(-1, 1)
    q_t1 = data['q_t1'].reshape(-1, 1)
    g = data['g'].reshape(-1, 1)
    t = data['t'].reshape(-1, 1)
    y = data['y'].reshape(-1, 1)
    x = data['x']

    return q_t0, q_t1, g, t, y, x


def get_estimate(q_t0, q_t1, g, t, y_dragon, truncate_level=0.01):
    """
    getting the back door adjustment & TMLE estimation
    """

    psi_n = psi_naive(q_t0, q_t1, g, t, y_dragon, truncate_level=truncate_level)
    psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(q_t0, q_t1, g, t,
                                                                                              y_dragon,
                                                                                              truncate_level=truncate_level)
    return psi_n, psi_tmle, initial_loss, final_loss, g_loss


def make_table(file_path='../../result/acic'):
    dict = {'tarnet': {'baseline': {'back_door': 0, }, 'targeted_regularization': 0},
            'dragonnet': {'baseline': 0, 'targeted_regularization': 0},
            'nednet': {'baseline': 0, 'targeted_regularization': 0}}

    tmle_dict = copy.deepcopy(dict)
    avg_dict = copy.deepcopy(dict)

    for knob in ['tarnet', 'dragonnet']:

        sim_dir = file_path + "/{}/".format(knob)

        ufids = sorted(glob.glob("{}/*".format(sim_dir)))
        for model in ['baseline', 'targeted_regularization']:
            ufid_simple = pd.Series(np.zeros(len(ufids)))
            ufid_tmle = pd.Series(np.zeros(len(ufids)))

            to_be_removed = []
            for i in range(len(ufids)):
                ufid = os.path.basename(ufids[i])
                usable = False
                for rep in range(5):
                    q_t0, q_t1, g, t, y, x = load_data(ufid=ufid, model=model, train_test='train', replication=rep,
                                                       file_path=sim_dir)
                    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, 0.05)
                    if y.size == 0:
                        usable = False
                    else:
                        usable = True
                        break
                if not usable:
                    to_be_removed.append(ufids[i])
            ufids = [ufid for ufid in ufids if ufid not in to_be_removed]
            for j in range(len(ufids)):
                ufid = os.path.basename(ufids[j])
                print(ufid)
                truth = load_truth(ufid=ufid)
                all_psi_n, all_psi_tmle = [], []
                for rep in range(5):
                    q_t0, q_t1, g, t, y, x = load_data(ufid=ufid, model=model, train_test='train', replication=rep,
                                                       file_path=sim_dir)
                    psi_n, psi_tmle, initial_loss, final_loss, g_loss = get_estimate(q_t0, q_t1, g, t, y,
                                                                                     truncate_level=0.01)
                    all_psi_n.append(psi_n)
                    all_psi_tmle.append(psi_tmle)

                err = abs(np.nanmean(all_psi_n) - truth)
                tmle_err = abs(np.nanmean(all_psi_tmle) - truth)

                ufid_simple[j] = err
                ufid_tmle[j] = tmle_err

            dict[knob][model] = ufid_simple.mean()
            tmle_dict[knob][model] = ufid_tmle.mean()
            avg_dict[knob][model] = ufid_tmle.to_numpy()

    base = avg_dict['tarnet']['baseline']
    base_treg = avg_dict['tarnet']['targeted_regularization']
    dragon = avg_dict['dragonnet']['baseline']
    dragon_treg = avg_dict['dragonnet']['targeted_regularization']


    bt = 0
    d = 0
    dt = 0
    for i in range(39):
        if base_treg[i] > base[i]:
            bt += 1
        if dragon[i] > base[i]:
            d += 1
        if dragon_treg[i] > base[i]:
            dt += 1

    print(f"bt: {bt/39}")
    print(f"d: {d / 39}")
    print(f"dt: {dt / 39}")

    return dict, tmle_dict


def main():
    dict, tmle_dict = make_table()
    print("The back door adjustment result is below")
    print(dict)

    print("the tmle estimator result is this ")
    print(tmle_dict)


if __name__ == "__main__":
    main()

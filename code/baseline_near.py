import dill
import numpy as np
from util import multi_label_metric, split_data, get_voc_data
import random
random.seed(1203)

def main():
    """
    Computes multi-label metrics and the DDI rate for a given set of test data.

    Returns:
        None
    """
    # Retrieve data
    data_train, data_eval, data_test = split_data()
    diag_voc, pro_voc, med_voc, voc_size = get_voc_data()
    ddi_adj_path = '../data/ddi_A_final.pkl'

    # Extract ground truth and predicted medication codes for each admission in test data
    gt = [patient[adm_idx+1][2] for patient in data_test if len(patient) > 1 for adm_idx, adm in enumerate(patient[:-1])]
    pred = [adm[2] for patient in data_test if len(patient) > 1 for adm_idx, adm in enumerate(patient[:-1])]

    # Compute multi-label metrics
    med_voc_size = len(med_voc.idx2word)
    y_gt = np.zeros((len(gt), med_voc_size))
    y_pred = np.zeros((len(gt), med_voc_size))
    for idx, code in enumerate(gt):
        y_gt[idx, code] = 1
    for idx, code in enumerate(pred):
        y_pred[idx, code] = 1
    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_gt, y_pred, y_pred)

    # DDI rate
    ddi_A = dill.load(open(ddi_adj_path, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    med_cnt = 0
    visit_cnt = 0

    for adm in y_pred:
        med_code_set = np.where(adm == 1)[0]
        visit_cnt += 1
        med_cnt += len(med_code_set)
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                    dd_cnt += 1
    
    ddi_rate = dd_cnt / all_cnt

    print('\tDDI Rate: %.4f, Jaccard: %.4f, PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1
    ))
    print('avg med', med_cnt/ visit_cnt)

if __name__ == '__main__':
    main()
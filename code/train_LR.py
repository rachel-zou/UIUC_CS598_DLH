import dill
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
import os
import random
random.seed(1203)
import itertools
from util import multi_label_metric, make_dir, split_data, get_voc_data, get_ehr_ddi_data

model_name = 'LR'

make_dir(model_name)

def create_dataset(data, diag_voc, pro_voc, med_voc):
    """
    Creates a dataset for multi-label classification from the input data and vocabulary objects.

    Args:
        data (list): A list of patients, where each patient is a list of visits, and each visit is a tuple of three lists containing the indices of the patient's diagnoses, procedures, and medications, respectively.
        diag_voc (Vocabulary): A Vocabulary object representing the diagnoses vocabulary.
        pro_voc (Vocabulary): A Vocabulary object representing the procedures vocabulary.
        med_voc (Vocabulary): A Vocabulary object representing the medications vocabulary.

    Returns:
        X (numpy.ndarray): A numpy array containing the input features of the dataset, where each row corresponds to a visit and each column corresponds to a diagnosis or procedure. The values are binary, indicating whether or not the visit includes the corresponding diagnosis or procedure.
        y (numpy.ndarray): A numpy array containing the output labels of the dataset, where each row corresponds to a visit and each column corresponds to a medication. The values are binary, indicating whether or not the medication was prescribed during the visit.
    """
    num_diags = len(diag_voc.idx2word)
    num_pros = len(pro_voc.idx2word)
    num_meds = len(med_voc.idx2word)
    num_inputs = num_diags + num_pros

    X = []
    y = []

    for patient in data:
        for visit in patient:
            diag_indices, pro_indices, med_indices = visit
            input_indices = diag_indices + [i + num_diags for i in pro_indices]

            multi_hot_input = np.zeros(num_inputs)
            multi_hot_input[input_indices] = 1

            multi_hot_output = np.zeros(num_meds)
            multi_hot_output[med_indices] = 1

            X.append(multi_hot_input)
            y.append(multi_hot_output)

    X = np.stack(X)
    y = np.stack(y)

    return X, y

def main():
    """
    Trains a logistic regression classifier on a multi-label classification dataset and computes multi-label metrics and the DDI rate.

    Returns:
        None
    """
    grid_search = False

    data_train, data_eval, data_test = split_data() 
    diag_voc, pro_voc, med_voc, voc_size = get_voc_data()
    

    train_X, train_y = create_dataset(data_train, diag_voc, pro_voc, med_voc)
    test_X, test_y = create_dataset(data_test, diag_voc, pro_voc, med_voc)
    eval_X, eval_y = create_dataset(data_eval, diag_voc, pro_voc, med_voc)

    if grid_search:
        params = {
            'estimator__penalty': ['l2'],
            'estimator__C': np.linspace(0.00002, 1, 100)
        }

        model = LogisticRegression()
        classifier = OneVsRestClassifier(model)
        lr_gs = GridSearchCV(classifier, params, verbose=1).fit(train_X, train_y)

        print("Best Params", lr_gs.best_params_)
        print("Best Score", lr_gs.best_score_)

        return

    
    model = LogisticRegression(C=0.90909)
    classifier = OneVsRestClassifier(model)
    classifier.fit(train_X, train_y)

    y_pred = classifier.predict(test_X)
    y_prob = classifier.predict_proba(test_X)

    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(test_y, y_pred, y_prob)

    # ddi rate
    ddi_A = dill.load(open('../data/ddi_A_final.pkl', 'rb'))
    all_cnt = 0
    dd_cnt = 0
    med_cnt = 0
    visit_cnt = 0

    for adm_idx, adm in enumerate(y_pred):
        meds = np.nonzero(adm)[0]
        visit_cnt += 1
        med_cnt += len(meds)
        for med_i, med_j in itertools.combinations(meds, 2):
            if med_j <= med_i:
                continue
            all_cnt += 1
            if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                dd_cnt += 1

    ddi_rate = dd_cnt / all_cnt

    print('\tDDI Rate: %.4f, Jaccard: %.4f, PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1
    ))

    history = defaultdict(list)
    for i in range(30):
        history['jaccard'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)

    dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

    print('avg med', med_cnt / visit_cnt)


if __name__ == '__main__':
    main()
import argparse
import os
import time
from collections import defaultdict

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from constants import RANDOM_SEED, EPOCH_NUM, LEARNING_RATE
from models import Retain
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, make_dir, get_voc_data, split_data

torch.manual_seed(RANDOM_SEED)

model_name = 'Retain'
resume_name = ''

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')

args = parser.parse_args()
model_name = args.model_name
resume_name = args.resume_path

def eval(model, data_eval, voc_size, epoch):
    """
    Evaluates the given model on the evaluation dataset.

    Args:
        model (Retain): The Retain model to be evaluated.
        data_eval (list): The list of patients in the evaluation dataset.
        voc_size (tuple): The tuple of vocabulary sizes for each data type.
        epoch (int): The current epoch number.

    Returns:
        (float, float, float, float, float, float, float): The ground truth DDI rate, predicted DDI rate, Jaccard index, 
                                                            PRAUC score, average precision, average recall, and average F1 score.
    """
    # evaluate
    print('')
    model.eval()

    smm_record = []
    gt_record = []  #RZ
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0

    for step, input in enumerate(data_eval):
        if len(input) < 2: # visit > 2
            continue

        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        y_gt_label = []  # RZ

        for i in range(1, len(input)):

            y_pred_label_tmp = []  #??

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[input[i][2]] = 1
            y_gt.append(y_gt_tmp)

            y_gt_label_tmp = np.where(y_gt_tmp == 1)[0] # RZ
            y_gt_label.append(sorted(y_gt_label_tmp))   # RZ

            target_output1 = model(input[:i])
            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)

            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp >= 0.3] = 1
            y_pred_tmp[y_pred_tmp < 0.3] = 0
            y_pred.append(y_pred_tmp)

            for idx, value in enumerate(y_pred_tmp):
                if value == 1:
                    y_pred_label_tmp.append(idx)
            y_pred_label.append(y_pred_label_tmp)

            med_cnt += len(y_pred_label_tmp)
            visit_cnt += 1

        smm_record.append(y_pred_label)
        gt_record.append(y_gt_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred),
                                                                                   np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient':input, 'y_label':y_pred_label}
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    dill.dump(case_study, open(os.path.join('saved', model_name, 'case_study.pkl'), 'wb'))
    # ddi rate
    ddi_rate = ddi_rate_score(smm_record)
    ddi_rate_gt = ddi_rate_score(gt_record)

    llprint('\tGT DDI Rate: %.4f, DDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate_gt, ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))
    print('avg med', med_cnt / visit_cnt)

    return ddi_rate_gt, ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def main():
    """
    Trains a Retain model using the specified hyperparameters and data, and saves the trained model and training
    history to disk. The Retain model is trained using the Adam optimizer with a binary cross-entropy loss function.
    The training is performed for a specified number of epochs, during which the model is evaluated on a validation set.
    If the model achieves a better performance on the validation set than in previous epochs, the model's weights are
    saved. Once the training is completed, the best-performing model is selected based on the evaluation on the validation
    set, and the model is evaluated on a test set. The model's final weights are saved to disk.

    Returns:
        None
    """
    make_dir(model_name)
    data_train, data_eval, data_test = split_data()
    diag_voc, pro_voc, med_voc, voc_size = get_voc_data()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    EPOCH = EPOCH_NUM   
    LR = LEARNING_RATE  
    TEST = args.eval

    model = Retain(voc_size, device=device)
    if TEST:
        model.load_state_dict(torch.load(open(os.path.join("saved", model_name, resume_name), 'rb')))

    model.to(device=device)
    print('parameters', get_n_params(model))


    optimizer = Adam(model.parameters(), lr=LR)

    if TEST:
        eval(model, data_test, voc_size, 0)
    else:
        history = defaultdict(list)
        best_epoch = 0
        best_ja = 0
        for epoch in range(EPOCH):
            loss_record = []
            start_time = time.time()
            model.train()
            for step, input in enumerate(data_train):
                if len(input) < 2:
                    continue

                loss = 0
                for i in range(1, len(input)):
                    target = np.zeros((1, voc_size[2]))
                    target[:, input[i][2]] = 1

                    output_logits = model(input[:i])
                    loss += F.binary_cross_entropy_with_logits(output_logits, torch.FloatTensor(target).to(device))
                    loss_record.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                llprint('\rTrain--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_train)))

            ddi_rate_gt, ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch)
            history['ja'].append(ja)
            history['ddi_rate'].append(ddi_rate)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss1: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))

            torch.save(model.state_dict(), open( os.path.join('saved', model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb'))
            print('')
            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja

        dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

        # test
        torch.save(model.state_dict(), open(
            os.path.join('saved', model_name, 'final_retain.model'), 'wb'))

        print('best_epoch:', best_epoch)


if __name__ == '__main__':
    main()
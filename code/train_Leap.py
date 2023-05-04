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
from models import Leap
from util import (
    ddi_rate_score, 
    get_n_params, 
    get_voc_data, 
    llprint,
    make_dir, 
    sequence_metric, 
    sequence_output_process,
    split_data
)

torch.manual_seed(RANDOM_SEED)

model_name = 'Leap'
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
    Evaluate a given model on the given evaluation data.

    Args:
        model (nn.Module): A PyTorch model.
        data_eval (list): A list of sequences, where each sequence is a list of admissions.
            Each admission is represented as a tuple (seq, seqlen, label), where:
            - seq (numpy array): A sequence of token IDs.
            - seqlen (int): The length of the sequence.
            - label (int): The label of the admission.
        voc_size (tuple): A tuple (num_tokens, num_classes, num_labels).
        epoch (int): The current epoch number.

    Returns:
        tuple: A tuple (ddi_rate_gt, ddi_rate, ja, prauc, avg_p, avg_r, avg_f1), where:
        - ddi_rate_gt (float): The ground truth DDI rate.
        - ddi_rate (float): The predicted DDI rate.
        - ja (float): The Jaccard similarity coefficient.
        - prauc (float): The area under the precision-recall curve.
        - avg_p (float): The average precision.
        - avg_r (float): The average recall.
        - avg_f1 (float): The average F1 score.
    """
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    records = []
    gt_record = [] #RZ
    med_cnt = 0
    visit_cnt = 0
    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        y_gt_label = []  # RZ
        for adm in input:
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            y_gt_label_tmp = np.where(y_gt_tmp == 1)[0] # RZ
            y_gt_label.append(sorted(y_gt_label_tmp))   # RZ

            output_logits = model(adm)
            output_logits = output_logits.detach().cpu().numpy()

            out_list, sorted_predict = sequence_output_process(output_logits, [voc_size[2], voc_size[2]+1])

            y_pred_label.append(sorted(sorted_predict))
            y_pred_prob.append(np.mean(output_logits[:, :-2], axis=0))

            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            visit_cnt += 1
            med_cnt += len(sorted_predict)
        records.append(y_pred_label)
        gt_record.append(y_gt_label) #RZ

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(records)
    ddi_rate_gt = ddi_rate_score(gt_record)
    llprint('\tGT DDI Rate: %.4f, DDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate_gt, ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))
    print('avg med', med_cnt / visit_cnt)
    return ddi_rate_gt, ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)

def main():
    """
    Trains a Leap model using the specified hyperparameters and data, and saves the trained model and training
    history to disk. The program creates a directory with the given `model_name`, splits the data into training,
    evaluation, and testing sets, and gets the vocabulary size of the data. It then sets the device to use for training
    (GPU if available, otherwise CPU), sets the number of epochs and learning rate, and defines the end token for the
    model. It initializes a Leap model with the given vocabulary size and device and loads the model state if testing
    mode is enabled. The program then prints the number of model parameters, initializes an Adam optimizer, and starts
    training. If testing mode is enabled, it evaluates the model on the test set, otherwise it trains the model for the
    given number of epochs and evaluates it on the evaluation set at each epoch. It records the evaluation metrics,
    saves the model state, and prints the results. Finally, it saves the history of the evaluation metrics and the final
    trained model.

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
    END_TOKEN = voc_size[2] + 1

    model = Leap(voc_size, device=device)
    if TEST:
        model.load_state_dict(torch.load(open(os.path.join("saved", model_name, resume_name), 'rb')))
        # pass

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
                for adm in input:
                    loss_target = adm[2] + [END_TOKEN]
                    output_logits = model(adm)
                    loss = F.cross_entropy(output_logits, torch.LongTensor(loss_target).to(device))

                    loss_record.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
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
            os.path.join('saved', model_name, 'final.model'), 'wb'))

        print('best_epoch:', best_epoch)

if __name__ == '__main__':
    main()
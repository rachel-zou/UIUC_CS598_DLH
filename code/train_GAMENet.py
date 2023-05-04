import argparse
import os
import time
from collections import defaultdict

import dill
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, RMSprop

from constants import *
from models import GAMENet
from util import (
    ddi_rate_score,
    get_ehr_ddi_data,
    get_n_params,
    get_voc_data,
    llprint,
    make_dir,
    multi_label_metric,
    split_data,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

model_name = 'GAMENet'
resume_name = ''

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')
parser.add_argument('--ddi', action='store_true', default=False, help="using ddi")

args = parser.parse_args()
model_name = args.model_name
resume_name = args.resume_path

def eval(model, data_eval, voc_size, epoch):
    """
    Evaluate the given model on the given evaluation data.

    Args:
        model (nn.Module): PyTorch model to be evaluated.
        data_eval (list): List of evaluation data.
        voc_size (tuple): Tuple of vocabulary sizes for diagnosis, procedure, and medication codes.
        epoch (int): Epoch number of the model being evaluated.

    Returns:
        tuple: A tuple of DDI rate ground truth, DDI rate, Jaccard score, PRAUC, AVG_PRC, AVG_RECALL, and AVG_F1.

    """

    model.eval()
    
    smm_record = []
    gt_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0

    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        y_gt_label = []  

        for adm_idx, adm in enumerate(input):

            target_output1 = model(input[:adm_idx+1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            y_gt_label_tmp = np.where(y_gt_tmp == 1)[0] 
            y_gt_label.append(sorted(y_gt_label_tmp))   

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)

            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))

            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)


        smm_record.append(y_pred_label)
        gt_record.append(y_gt_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        
        case_study[adm_ja] = {
            'ja': adm_ja, 
            'patient': input, 
            'y_label': y_pred_label,
        }

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record)
    ddi_rate_gt = ddi_rate_score(gt_record)

    llprint('\tGT DDI Rate: %.4f, DDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate_gt, ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))
    dill.dump(obj=smm_record, file=open('../data/gamenet_records.pkl', 'wb'))
    dill.dump(case_study, open(os.path.join('saved', model_name, 'case_study.pkl'), 'wb'))

    print('avg med', med_cnt / visit_cnt)

    return ddi_rate_gt, ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def main():
    """
    Trains and evaluates a GAMENet model on EHR data for disease diagnosis and drug interaction prediction.
    It creates a directory with the name provided as an argument to the function.
    It loads and prepares the data required for the training process. This includes the EHR and DDI data, vocabulary data, and data split into train, evaluation, and test sets.
    It initializes the model and sets the hyperparameters such as learning rate, number of epochs, decay weight, and temperature for simulated annealing.
    It trains the model for the specified number of epochs using the Adam optimizer and binary cross-entropy and multilabel margin loss functions. It also implements negative sampling for DDI prediction and annealing for simulated annealing during the training process.
    After each epoch, the model is evaluated on the evaluation set, and the performance metrics such as Jaccard similarity index, DDI rate, average precision, recall, F1 score, and precision-recall AUC are calculated and stored in the history dictionary.
    The trained model is saved at the end of each epoch and the best epoch is identified based on the Jaccard similarity index. Finally, the history of the training process is saved using the dill library.
    
    Returns:
        None
    """
    make_dir(model_name)

    data_train, data_eval, data_test = split_data()
    diag_voc, pro_voc, med_voc, voc_size = get_voc_data()
    ehr_adj, ddi_adj = get_ehr_ddi_data()   

    EPOCH = EPOCH_NUM   
    LR = LEARNING_RATE  
    TEST = args.eval
    Neg_Loss = args.ddi
    DDI_IN_MEM = args.ddi
    TARGET_DDI = TARGET_DDI_VAL
    T = T_VAL
    decay_weight = DECAY_WEIGHT_VAL
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = GAMENet(voc_size, ehr_adj, ddi_adj, emb_dim=64, device=device, ddi_in_memory=DDI_IN_MEM)
    if TEST:
        model.load_state_dict(torch.load(open(resume_name, 'rb')))
    model.to(device=device)

    print('parameters', get_n_params(model))
    optimizer = RMSprop(list(model.parameters()), lr=LR)

    if TEST:
        eval(model, data_test, voc_size, 0)   
    else:
        history = defaultdict(list)
        best_epoch = 0
        best_ja = 0
        for epoch in range(EPOCH):
            loss_record1 = []
            start_time = time.time()
            model.train()
            prediction_loss_cnt = 0
            neg_loss_cnt = 0
            for step, input in enumerate(data_train):
                for idx, adm in enumerate(input):
                    seq_input = input[:idx+1]
                    loss1_target = np.zeros((1, voc_size[2]))
                    loss1_target[:, adm[2]] = 1
                    loss3_target = np.full((1, voc_size[2]), -1)
                    for idx, item in enumerate(adm[2]):
                        loss3_target[0][idx] = item

                    target_output1, batch_neg_loss = model(seq_input)

                    loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))
                    loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))
                    if Neg_Loss:
                        target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
                        target_output1[target_output1 >= 0.5] = 1
                        target_output1[target_output1 < 0.5] = 0
                        y_label = np.where(target_output1 == 1)[0]
                        current_ddi_rate = ddi_rate_score([[y_label]])
                        if current_ddi_rate <= TARGET_DDI:
                            loss = 0.9 * loss1 + 0.01 * loss3
                            prediction_loss_cnt += 1
                        else:
                            rnd = np.exp((TARGET_DDI - current_ddi_rate)/T)
                            if np.random.rand(1) < rnd:
                                loss = batch_neg_loss
                                neg_loss_cnt += 1
                            else:
                                loss = 0.9 * loss1 + 0.01 * loss3
                                prediction_loss_cnt += 1
                    else:
                        loss = 0.9 * loss1 + 0.01 * loss3

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record1.append(loss.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (epoch, step, len(data_train), prediction_loss_cnt, neg_loss_cnt))
            # annealing
            T *= decay_weight

            ddi_rate_gt, ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch)

            history['ja'].append(ja)
            history['ddi_rate'].append(ddi_rate)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
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

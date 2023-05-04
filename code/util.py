from sklearn.metrics import roc_auc_score, precision_score, average_precision_score
import numpy as np
import pandas as pd
import sys
import warnings
import dill
import os
import random
from collections import Counter
warnings.filterwarnings('ignore')

def make_dir(model_name):
    """
    Creates a directory with the given name in the "saved" directory if it does not already exist.

    Args:
        model_name (str): the name of the directory to create.

    Returns:
        None
    """
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

def split_data(random_seed=1203):
    """
    Splits the dataset into train, evaluation, and test sets.

    Args:
        random_seed (int): the seed for the random number generator used to shuffle the dataset.

    Returns:
        data_train (list): a list of training examples.
        data_eval (list): a list of evaluation examples.
        data_test (list): a list of test examples.
    """
    data_path = '../data/records_final.pkl'
    data = dill.load(open(data_path, 'rb'))

    random.seed(random_seed)
    split_point = int(len(data) * 2 / 3)
    random.shuffle(data)  
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    return data_train, data_eval, data_test

def get_voc_data():
    """
    Load the vocabularies for the diagnosis codes, procedure codes and medication codes, and return them along with the
    size of each vocabulary as a tuple.

    Returns:
        diag_voc (Vocabulary): the vocabulary for diagnosis codes.
        pro_voc (Vocabulary): the vocabulary for procedure codes.
        med_voc (Vocabulary): the vocabulary for medication codes.
        voc_size (tuple): a tuple containing the size of each vocabulary in the order (diag_voc_size, pro_voc_size, med_voc_size).
    """
    voc_path = '../data/voc_final.pkl'
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    return diag_voc, pro_voc, med_voc, voc_size

def get_ehr_ddi_data():
    """
    Load the EHR and DDI adjacency matrices from the respective pickle files.
    
    Returns:
        ehr_adj (np.ndarray): the EHR adjacency matrix.
        ddi_adj (np.ndarray): the DDI adjacency matrix.
    """
    ehr_adj_path = '../data/ehr_adj_final.pkl'
    ddi_adj_path = '../data/ddi_A_final.pkl'
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    return ehr_adj, ddi_adj

def get_n_params(model):
    """
    Get the total number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): the PyTorch model for which to count the parameters.

    Returns:
        n_params (int): the total number of trainable parameters in the model.
    """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def llprint(message):
    """
    Prints the input message to the console and flushes the standard output buffer.
    
    Args:
        message (str): the message to be printed.
    """
    sys.stdout.write(message)
    sys.stdout.flush()

def sequence_output_process(output_logits, filter_token):
    """
    Process output logits from a sequence prediction model, filtering out tokens from a set of forbidden tokens.

    Args:
        output_logits (numpy array): A 2D numpy array of shape (batch_size, num_classes) containing the output logits 
                                     from the model.
        filter_token (set): A set of integers representing the forbidden tokens that should be filtered out.

    Returns:
        out_list (list): A list of integers representing the predicted output sequence, after filtering out the tokens 
                         in filter_token.
        sorted_predict (list): A list of integers representing the predicted output sequence, sorted in descending 
                               order of probability.
    """
    sorted_indices = np.argsort(output_logits, axis=-1)[:, ::-1]
    out_list = []
    break_flag = False
    
    for i in range(sorted_indices.shape[0]):
        if break_flag:
            break
        
        for j in range(sorted_indices.shape[1]):
            label = sorted_indices[i][j]
            
            if label in filter_token:
                break_flag = True
                break
                
            if label not in out_list:
                out_list.append(label)
                break
                
    y_pred_prob_tmp = output_logits[np.arange(len(out_list)), out_list]
    sorted_predict = [x for _, x in sorted(zip(y_pred_prob_tmp, out_list), reverse=True)]
    
    return out_list, sorted_predict

def sequence_metric(y_gt, y_pred, y_prob, y_label):
    """
    Computes evaluation metrics for a sequence tagging task.

    Args:
        y_gt (np.array): Ground truth labels with shape (batch_size, seq_len, n_classes).
        y_pred (np.array): Predicted labels with shape (batch_size, seq_len, n_classes).
        y_prob (np.array): Predicted probabilities with shape (batch_size, seq_len, n_classes).
        y_label (list): Predicted labels (indices) after processing the output logits, with shape (batch_size, max_seq_len).

    Returns:
        ja (float): Jaccard similarity score averaged over all batches.
        prauc (float): Precision-Recall AUC score averaged over all batches.
        avg_prc (float): Average precision score averaged over all batches.
        avg_recall (float): Average recall score averaged over all batches.
        avg_f1 (float): Average F1 score averaged over all batches.
    """
    def average_prc(y_gt, y_label):
        """
        Computes average precision score for each sample in the batch.

        Args:
            y_gt (np.array): Ground truth labels in binary matrix format of shape (batch_size, num_classes)
            y_label (list): Predicted labels as a list of lists where each sublist contains the indices of predicted labels
                            for that sample.

        Returns:
            np.array: Array of average precision scores for each sample in the batch.
        """
        scores = []
        for i in range(y_gt.shape[0]):
            targets = np.where(y_gt[i] == 1)[0]
            out_list = y_label[i]
            intersection = set(out_list).intersection(set(targets))
            precision = len(intersection) / len(out_list) if len(out_list) > 0 else 0
            scores.append(precision)
        return scores

    def average_recall(y_gt, y_label):
        """
        Computes the average recall score.

        Parameters:
            y_gt (numpy array): Ground truth label matrix with shape (num_samples, num_classes).
            y_label (list): Predicted labels list for each sample with shape (num_samples, num_classes).

        Returns:
            numpy array: Average recall score for each sample with shape (num_samples,).
        """
        scores = []
        for i in range(y_gt.shape[0]):
            target = set(np.where(y_gt[i] == 1)[0])
            out_list = set(y_label[i])
            inter = out_list.intersection(target)
            recall_score = len(inter) / len(target) if len(target) > 0 else 0
            scores.append(recall_score)
        return scores

    def average_f1(precision_scores, recall_scores):
        """
        Calculates the F1 score for each pair of precision and recall scores.

        Args:
            precision_scores: A list of float values representing the precision scores for each sample.
            recall_scores: A list of float values representing the recall scores for each sample.

        Returns:
            f1_scores: A list of float values representing the F1 score for each pair of precision and recall scores.
        """
        f1_scores = []
        for i in range(len(precision_scores)):
            if precision_scores[i] + recall_scores[i] == 0:
                f1_scores.append(0)
            else:
                f1_scores.append(2 * precision_scores[i] * recall_scores[i] / (precision_scores[i] + recall_scores[i]))
        return f1_scores

    def jaccard(y_gt, y_label):
        """
        Calculate the Jaccard similarity score between the predicted labels and the ground truth labels.

        Args:
            y_gt (numpy.ndarray): An array containing the ground truth labels, with shape (n_samples, n_classes).
            y_label (numpy.ndarray): An array containing the predicted labels, with shape (n_samples, n_classes).

        Returns:
            float: The mean Jaccard similarity score across all samples.
        """
        score = []
        for b in range(y_gt.shape[0]):
            target = set(np.where(y_gt[b] == 1)[0])
            out_list = set(y_label[b])
            if len(target) == 0 and len(out_list) == 0:
                score.append(1)
            else:
                inter = out_list.intersection(target)
                union = out_list.union(target)
                jaccard_score = len(inter) / len(union)
                score.append(jaccard_score)
        return np.mean(score)

    def roc_auc(y_gt, y_pred_prob):
        """
        Calculate the mean ROC-AUC score across all classes for a given set of ground truth labels and predicted probabilities.

        Args:
            y_gt (numpy.ndarray): A binary matrix of shape (n_samples, n_classes) representing the ground truth labels.
            y_pred_prob (numpy.ndarray): A matrix of shape (n_samples, n_classes) representing the predicted probabilities.

        Returns:
            float: The mean ROC-AUC score across all classes.
        """
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_pred_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        """
        Computes the macro-averaged area under the precision-recall curve (AUC-PR) for a set of predicted probability
        scores and ground truth labels.

        Args:
            y_gt (numpy.ndarray): An array of shape (n_samples, n_classes) representing the ground truth labels,
                where each row corresponds to a sample and each column corresponds to a binary label.
            y_prob (numpy.ndarray): An array of shape (n_samples, n_classes) representing the predicted probability
                scores, where each row corresponds to a sample and each column corresponds to a probability score.

        Returns:
            float: The macro-averaged AUC-PR score for the given predicted probability scores and ground truth labels.
        """
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    try:
        auc = roc_auc(y_gt, y_prob)
    except ValueError:
        auc = 0

    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def multi_label_metric(y_gt, y_pred, y_prob):
    """
    Compute various multi-label classification metrics between ground truth labels and predicted labels.

    Args:
        y_gt (numpy.ndarray): Ground truth labels with shape (num_samples, num_labels).
        y_pred (numpy.ndarray): Predicted binary labels with shape (num_samples, num_labels).
        y_prob (numpy.ndarray): Predicted probabilities with shape (num_samples, num_labels).

    Returns:
        ja (float): Mean Jaccard similarity coefficient across all labels.
        prauc (float): Mean area under the precision-recall curve (PR AUC) across all labels.
        avg_prc (float): Mean precision across all labels.
        avg_recall (float): Mean recall across all labels.
        avg_f1 (float): Mean F1 score across all labels.
    """
    def jaccard(y_gt, y_pred):
        """
        Compute Jaccard similarity coefficient between ground truth and predicted labels.

        Args:
            y_gt (numpy.ndarray): Ground truth labels with shape (num_labels,).
            y_pred (numpy.ndarray): Predicted binary labels with shape (num_labels,).

        Returns:
            score (float): Jaccard similarity coefficient.
        """
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        """
        Compute average precision between ground truth and predicted labels.

        Args:
            y_gt (numpy.ndarray): Ground truth labels with shape (num_labels,).
            y_pred (numpy.ndarray): Predicted binary labels with shape (num_labels,).

        Returns:
            score (float): Average precision.
        """
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        """
        Compute average recall between ground truth and predicted labels.

        Args:
            y_gt (numpy.ndarray): Ground truth labels with shape (num_labels,).
            y_pred (numpy.ndarray): Predicted binary labels with shape (num_labels,).

        Returns:
            score (float): Average recall.
        """
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        """
        Calculates the F1 score for each pair of precision and recall scores.

        Args:
            average_prc: A list of float values representing the precision scores for each sample.
            average_recall: A list of float values representing the recall scores for each sample.

        Returns:
            score: A list of float values representing the F1 score for each pair of precision and recall scores.
        """
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def roc_auc(y_gt, y_prob):
        """
        Computes the macro-averaged ROC AUC score of a set of predicted probabilities.

        Args:
            y_gt : array-like of shape (n_samples, n_classes)
                Ground-truth binary labels. Each row corresponds to a sample, and each column
                corresponds to a binary class.
            y_prob : array-like of shape (n_samples, n_classes)
                Predicted probabilities. Each row corresponds to a sample, and each column
                corresponds to the predicted probability of the positive class.

        Returns:
            float: The macro-averaged ROC AUC score of the predicted probabilities.
        """
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        """
        Computes the macro-averaged precision-recall AUC score of a set of predicted probabilities.

        Args:
            y_gt : array-like of shape (n_samples, n_classes)
                Ground-truth binary labels. Each row corresponds to a sample, and each column
                corresponds to a binary class.
            y_prob : array-like of shape (n_samples, n_classes)
                Predicted probabilities. Each row corresponds to a sample, and each column
                corresponds to the predicted probability of the positive class.

        Returns:
            float: The macro-averaged precision-recall AUC score of the predicted probabilities.
        """
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    auc = roc_auc(y_gt, y_prob)

    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_pred)
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def ddi_rate_score(record, path='../data/ddi_A_final.pkl'):
    """
    Computes the rate of drug-drug interactions (DDIs) in a patient record using a pre-computed DDI matrix.

    Args:
        record : list of list of int
            A patient record represented as a list of admissions, where each admission is represented
            as a list of medication codes. Medication codes are integers that index into the DDI matrix.
        path : str, optional
            The file path to the pre-computed DDI matrix in pickle format. The default is '../data/ddi_A_final.pkl'.

    Returns:
        float
            The rate of DDIs in the patient record, defined as the number of pairs of drugs that interact
            divided by the total number of drug pairs considered.
    """
    # Load ddi matrix
    ddi_A = dill.load(open(path, 'rb'))

    # Count drug pairs that interact
    all_cnt, dd_cnt = 0, 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j in range(i + 1, len(med_code_set)):
                    all_cnt += 1
                    if ddi_A[med_i, med_code_set[j]] == 1:
                        dd_cnt += 1
                        
    # Compute ddi rate score
    if all_cnt == 0:
        return 0
    else:
        return dd_cnt / all_cnt
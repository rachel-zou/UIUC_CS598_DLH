# Final Project for UIUC CS598 DLH

Today abundant health data such as electronic health records (EHR) supports the development of better computational models. Deep learning methods have been widely utilized to design medication recommendation algorithms but none of them considers the adverse drug-drug interactions (DDI). 

For our final project in CS598 Deep Learning in Healthcare, we attempted to reproduce the research paper titled "GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination", which proposed the Graph Augmented Memory Networks (GAMENet) to fill this gap. This deep learning model takes both longitudinal patient EHR data and DDIs based knowledge as inputs and outputs recommendation of medication combinations.

### The code folder contains the following scripts:
 - util.py - Helper functions used in the models.
 - layers.py - Simple GCN layer.
 - models.py - Deep learning models.
 - constants.py - Parameters used in the models.
 - preprocessing.py - Data preprocessing.
 - baseline_near.py - Recommend the same combination medications at previous visit for current visit.
 - train_LR.py - Use a logistic regression with L2 regularization to make the predictions.
 - train_Retain.py - A two-level neural attention model which detects influential past visits and significant clinical variables within those visits. 
 - train_Leap.py - Leap formulates a multi-instance multi-label learning framework and proposes a variant of sequence-to-sequence model based on content-attention mechanism to predict combination of medicines given patient’s diagnoses.
 - train_GAMENet.py - Integrates the drug-drug interactions knowledge graph by a memory module implemented as a graph convolutional networks, and models longitudinal patient records as the query. 

### The data folder contains the following files:
 - DLH Final Project Demo Notebook.ipynb - The demo notebook we created for the final project.
 - drug-atc.csv - Mapping files for drug code transformation.
 - ndc2atc_level4.csv - Mapping files for drug code transformation.
 - ndc2rxnorm_mapping.txt - Mapping files for drug code transformation.
 - data_final.pkl - Final clean data for the models (deleted on May 16, 2023 per TA's requirement).
 - ddi_A_final.pkl - Drug-drug adjacency matrix constructed from DDI dataset.
 - ehr_adj_final.pkl - Drug-drug adjacency matrix constructed from EHR dataset.
 - records_final.pkl - Input data with four dimensions (patient_idx, visit_idx, medical modal, medical id) where medical model equals 3 made of diagnosis, procedure and drug (deleted on May 16, 2023 per TA's requirement).
 - voc_final.pkl - The vocabulary list to transform medical word to corresponding idx.
 
 DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv and drug-DDI.csv are necessary files. They are not included here due to the size limit, but can be downloaded from [MIMIC](https://mimic.mit.edu/docs/gettingstarted/) and [DDI data](https://www.dropbox.com/s/8os4pd2zmp2jemd/drug-DDI.csv?dl=0).
 
### Base models:
 
python baseline_near.py

python train_LR.py

python train_Leap.py --model_name Leap 

python train_Leap.py --model_name Leap --resume_path Epoch_{}_JA_{}_DDI_{}.model --eval 

python train_Retain.py --model_name Retain

python train_Retain.py --model_name Retain --resume_path Epoch_{}_JA_{}_DDI_{}.model --eval 

### GAMENet

python train_GAMENet.py --model_name GAMENet --ddi # training with DDI knowledge

python train_GAMENet.py --model_name GAMENet --ddi --resume_path Epoch_{}_JA_{}_DDI_{}.model --eval # testing with DDI knowledge

python train_GAMENet.py --model_name GAMENet # training without DDI knowledge

python train_GAMENet.py --model_name GAMENet --resume_path Epoch_{}_JA_{}_DDI_{}.model --eval # testing with DDI knowledge

### Below is the original paper information. 

@article{shang2018gamenet,
  title="{GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination}",
  author={Shang, Junyuan and Xiao, Cao and Ma, Tengfei and Li, Hongyan and Sun, Jimeng},
  journal={arXiv preprint arXiv:1809.01852},
  year={2018}
}

The original code can be found [here](https://github.com/sjy1203/GAMENet). 

import pandas as pd
import functools
import dill
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Read from MIMIC csv files
# Files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
med_file = 'PRESCRIPTIONS.csv'
diag_file = 'DIAGNOSES_ICD.csv'
procedure_file = 'PROCEDURES_ICD.csv'

# drug code mapping files (already in ./data/)
ndc2atc_file = 'ndc2atc_level4.csv' 
cid_atc = 'drug-atc.csv'
ndc_rxnorm_file = 'ndc2rxnorm_mapping.txt'

# drug-drug interactions can be down https://www.dropbox.com/s/8os4pd2zmp2jemd/drug-DDI.csv?dl=0
ddi_file = 'drug-DDI.csv'

def convert_to_list(x):
    """
    This function takes an input `x` and converts it into a list.
    
    Args:
        x (str or iterable): The input to be converted into a list.
    
    Returns:
        list: The input `x` as a list. If `x` is already a list, it is returned as is.
    """
    if isinstance(x, str):
        return [x]
    else:
        return list(x)

def process_procedure(procedure_file):
    """
    Read and process the procedure CSV file.
    
    Args:
        procedure_file: str, the path to the CSV file containing procedure data
    
    Returns:
        pro_pd: pandas DataFrame, the processed procedure data
    """
    # Read the CSV file and set data types
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE':'category'})

    # Drop unnecessary columns and duplicates
    pro_pd = pro_pd.drop(columns=['ROW_ID', 'SEQ_NUM']).drop_duplicates()

    # Sort the data by SUBJECT_ID, HADM_ID, and ICD9_CODE
    pro_pd = pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'])

    # Reset the index
    pro_pd = pro_pd.reset_index(drop=True)

    return pro_pd

def process_med(med_file):
    """
    Processes medication data from a CSV file and returns a cleaned and filtered pandas DataFrame.

    Args:
        med_file (str): A string specifying the path and file name of the CSV file containing medication data.

    Returns:
        pandas DataFrame: A cleaned and filtered pandas DataFrame containing medication data.
    """
    med_pd = pd.read_csv(med_file, dtype={'NDC': 'category'})
    
    # filter and clean data
    med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
                          'FORMULARY_DRUG_CD','GSN','PROD_STRENGTH','DOSE_VAL_RX',
                          'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP',
                          'ROUTE','ENDDATE','DRUG'], inplace=True)
    med_pd.drop(med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')    
    med_pd.drop_duplicates(inplace=True)
    
    # sort by columns
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd.reset_index(drop=True, inplace=True)
    
    def filter_first24hour_med(med_pd):
        """
        Filter medication data to keep only the first ICU stay for each patient and merge with original dataframe to keep the NDC code.

        Args:
            med_pd: pandas.DataFrame
                The medication data to be processed.

        Returns:
            pandas.DataFrame
                The processed medication data with only the first ICU stay for each patient and the NDC code.
        """
        # Keep only the first ICU stay for each patient
        med_pd_new = med_pd.drop(columns=['NDC'])
        med_pd_new = med_pd_new.drop_duplicates(subset=['SUBJECT_ID','HADM_ID','ICUSTAY_ID'], keep='first')

        # Merge with original dataframe to keep the NDC code
        med_pd_new = pd.merge(med_pd_new, med_pd, on=['SUBJECT_ID','HADM_ID','ICUSTAY_ID','STARTDATE'])
        med_pd_new = med_pd_new.drop(columns=['STARTDATE'])

        return med_pd_new

    med_pd = filter_first24hour_med(med_pd)

    # Drop the 'ICUSTAY_ID' column from med_pd
    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])

    # Drop duplicates from med_pd
    med_pd = med_pd.drop_duplicates().reset_index(drop=True)
    
    # visit > 2
    def process_visit_lg2(med_pd):
        """
        Filters `med_pd` dataframe to include only the first ICU stay for each patient with more than one visit.
        
        Args:
            med_pd : pandas.DataFrame
                The input DataFrame containing medication data.
            
        Returns:
            pandas.DataFrame
                A new DataFrame with the same columns as `med_pd`, but only including rows for the first ICU stay of each
                patient with more than one visit. If a patient has only one ICU stay, all rows for that patient are included.
                Additionally, a new column 'HADM_ID_Len' is added which contains the number of HADM_IDs per SUBJECT_ID.
        """
        # Get unique HADM_IDs per SUBJECT_ID
        unique_hadm_ids = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
        subject_counts = unique_hadm_ids['SUBJECT_ID'].value_counts()

        # Filter SUBJECT_IDs with visit counts > 1
        subjects_with_multiple_visits = subject_counts[subject_counts > 1].index
        visits_lg2 = unique_hadm_ids[unique_hadm_ids['SUBJECT_ID'].isin(subjects_with_multiple_visits)]

        # Add HADM_ID count as a new column
        visits_lg2['HADM_ID_Len'] = visits_lg2.groupby('SUBJECT_ID')['HADM_ID'].transform('count')

        # Return result
        return visits_lg2

    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)    
    med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')    
    
    return med_pd.reset_index(drop=True)

def process_diag(diag_file):
    """
    Reads in the diagnosis data from the provided file, removes any rows with missing data, unnecessary columns,
    and duplicate rows. The resulting DataFrame is sorted by subject ID and hospital admission ID and the index is reset.
    
    Args:
        diag_file: string, path to the diagnosis data file
    
    Returns:
        diag_pd: pandas DataFrame, diagnosis data processed and sorted by subject ID and hospital admission ID
    """
    # Read in the diagnosis data from the provided file
    diag_pd = pd.read_csv(diag_file)
    
    # Drop any rows with missing data
    diag_pd.dropna(inplace=True)
    
    # Remove unnecessary columns
    diag_pd.drop(columns=['SEQ_NUM','ROW_ID'], inplace=True)
    
    # Drop any duplicate rows
    diag_pd.drop_duplicates(inplace=True)
    
    # Sort the data by subject ID and hospital admission ID
    diag_pd.sort_values(by=['SUBJECT_ID','HADM_ID'], inplace=True)
    
    # Reset the index of the DataFrame and return it
    return diag_pd.reset_index(drop=True)

def ndc2atc4(med_pd):
    """
    Converts the NDC codes in the medication DataFrame `med_pd` to ATC4 codes using external data files.

    Args:
        med_pd (pandas DataFrame): DataFrame containing medication data with NDC codes.

    Returns:
        pandas DataFrame: DataFrame containing medication data with ATC4 codes.
    """
    with open(ndc_rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(subset=['RXCUI'], inplace=True)

    rxnorm2atc = pd.read_csv(ndc2atc_file, usecols=['RXCUI', 'ATC4'], squeeze=True)
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)

    med_pd = med_pd[~med_pd['RXCUI'].isin([''])]
    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')

    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)

    med_pd['NDC'] = med_pd['ATC4'].str[:4]
    med_pd.drop_duplicates(inplace=True)
    med_pd.reset_index(drop=True, inplace=True)

    return med_pd

def filter_2000_most_diag(diag_pd):
    """
    Filter the diagnosis DataFrame to keep only the rows with the top 2000 most frequent ICD9 codes.
    
    Args:
        diag_pd : pandas.DataFrame
            DataFrame containing diagnosis data with 'ICD9_CODE' column.
    
    Returns:
        pandas.DataFrame
            DataFrame with only the rows containing ICD9 codes in the top 2000 most frequent list.
    """
    # Get the top 2000 most frequent ICD9 codes
    diag_count = diag_pd['ICD9_CODE'].value_counts().reset_index().rename(columns={'index': 'ICD9_CODE', 'ICD9_CODE': 'count'})
    top_2000_icd9_codes = diag_count.loc[:1999, 'ICD9_CODE'].tolist()
    
    # Filter the DataFrame to keep only the rows with ICD9 codes in the top 2000
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(top_2000_icd9_codes)]
    
    return diag_pd.reset_index(drop=True)

def process_all():
    """
    This function processes medication, diagnosis, and procedure data and returns a pandas DataFrame with the processed data.
    
    Returns:
        data (pandas DataFrame): A DataFrame containing processed medication, diagnosis, and procedure data, merged on unique SUBJECT_ID and HADM_ID. The DataFrame has the following columns:
            - SUBJECT_ID: unique identifier for each patient
            - HADM_ID: unique identifier for each hospital admission
            - ICD9_CODE: list of diagnosis codes associated with the admission
            - NDC: list of medication NDC codes associated with the admission
            - PRO_CODE: list of procedure codes associated with the admission
            - NDC_Len: number of unique NDC codes associated with the admission
    """
    # get med and diag (visit>=2)
    medication_pd = process_med(med_file)
    medication_pd = ndc2atc4(medication_pd)

    diagnosis_pd = process_diag(diag_file)
    diagnosis_pd = filter_2000_most_diag(diagnosis_pd)

    procedure_pd = process_procedure(procedure_file)

    medication_pd_key = medication_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diagnosis_pd_key = diagnosis_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    procedure_pd_key = procedure_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    combined_key = functools.reduce(lambda x, y: pd.merge(x, y, on=['SUBJECT_ID', 'HADM_ID'], how='inner'), 
                                    [medication_pd_key, diagnosis_pd_key, procedure_pd_key])
    
    diagnosis_pd = diagnosis_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    medication_pd = medication_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    procedure_pd = procedure_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # flatten and merge
    diagnosis_pd = diagnosis_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index()  
    medication_pd = medication_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
    procedure_pd = procedure_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})  
    medication_pd['NDC'] = medication_pd['NDC'].map(convert_to_list)
    procedure_pd['PRO_CODE'] = procedure_pd['PRO_CODE'].map(convert_to_list)
    
    data = functools.reduce(lambda x, y: pd.merge(x, y, on=['SUBJECT_ID', 'HADM_ID'], how='inner'), 
                            [diagnosis_pd, medication_pd, procedure_pd])
    
    data['NDC_Len'] = data['NDC'].map(len)
    return data

def statistics():
    """
    This function prints various statistics related to the processed medical data. These statistics include the number of unique patients and clinical events, the number of unique diagnoses, medications, and procedures, and various averages and maxes calculated over patient visits.

        Prints:
        - #patients: number of unique patients
        - #clinical events: total number of clinical events
        - #diagnosis: number of unique diagnosis codes
        - #med: number of unique medication NDC codes
        - #procedure: number of unique procedure codes
        - #avg of diagnoses: average number of diagnoses per clinical event
        - #avg of medicines: average number of medications per clinical event
        - #avg of procedures: average number of procedures per clinical event
        - #avg of vists: average number of visits per patient
        - #max of diagnoses: maximum number of diagnoses associated with a single visit
        - #max of medicines: maximum number of medications associated with a single visit
        - #max of procedures: maximum number of procedures associated with a single visit
        - #max of visit: maximum number of visits associated with a single patient
    """
    print('#patients ', data['SUBJECT_ID'].nunique())
    print('#clinical events ', len(data))
    
    unique_diag = data['ICD9_CODE'].explode().nunique()
    unique_med = data['NDC'].explode().nunique()
    unique_pro = data['PRO_CODE'].explode().nunique()
    
    print('#diagnosis ', unique_diag)
    print('#med ', unique_med)
    print('#procedure', unique_pro)
    
    avg_diag = 0
    avg_med = 0
    avg_pro = 0
    max_diag = 0
    max_med = 0
    max_pro = 0
    cnt = 0
    max_visit = 0
    avg_visit = 0

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]
        x = []
        y = []
        z = []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['ICD9_CODE']))
            y.extend(list(row['NDC']))
            z.extend(list(row['PRO_CODE']))
        x = set(x)
        y = set(y)
        z = set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y) 
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt
        
    print('#avg of diagnoses ', avg_diag / cnt)
    print('#avg of medicines ', avg_med / cnt)
    print('#avg of procedures ', avg_pro / cnt)
    print('#avg of vists ', avg_visit / len(data['SUBJECT_ID'].unique()))

    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of procedures ', max_pro)
    print('#max of visit ', max_visit)
    
    
data = process_all()
statistics()
data.to_pickle('data_final.pkl')

# Create vocabulary for medical codes and save patient record in pickle form
class Voc:
    """A vocabulary class that maps words to indices and vice versa."""
    
    def __init__(self):
        """Initialize the vocabulary with empty word-to-index and index-to-word mappings."""
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        """Add a sentence to the vocabulary.
        
        Args:
            sentence (list): A list of words in the sentence.
        """
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)
                
def create_str_token_mapping(df):
    """Create string to token mappings for diagnosis, medication, and procedure codes.
    
    Args:
        df (pandas.DataFrame): A DataFrame containing the medical records data.
        
    Returns:
        Tuple[Voc, Voc, Voc]: A tuple of three Voc objects, one for each code type.
    """
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()
    
    for index, row in df.iterrows():
        diag_voc.add_sentence(row['ICD9_CODE'])
        med_voc.add_sentence(row['NDC'])
        pro_voc.add_sentence(row['PRO_CODE'])
    
    with open('voc_final.pkl', 'wb') as f:
        dill.dump({'diag_voc': diag_voc, 'med_voc': med_voc, 'pro_voc': pro_voc}, f)
        
    return diag_voc, med_voc, pro_voc

def create_patient_record(df, diag_voc, med_voc, pro_voc):
    """Create a patient record data structure from the medical records data.
    
    Args:
        df (pandas.DataFrame): A DataFrame containing the medical records data.
        diag_voc (Voc): A Voc object for diagnosis codes.
        med_voc (Voc): A Voc object for medication codes.
        pro_voc (Voc): A Voc object for procedure codes.
        
    Returns:
        list: A list of patient records, where each record is a list of admissions, and each admission is a
        list of three lists of code indices (one for each code type).
    """
    records = [] # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['ICD9_CODE']])
            admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2idx[i] for i in row['NDC']])
            patient.append(admission)
        records.append(patient) 
    
    with open('records_final.pkl', 'wb') as f:
        dill.dump(records, f)
        
    return records
        
    
path='data_final.pkl'
df = pd.read_pickle(path)
diag_voc, med_voc, pro_voc = create_str_token_mapping(df)
records = create_patient_record(df, diag_voc, med_voc, pro_voc)
print(len(diag_voc.idx2word), len(med_voc.idx2word), len(pro_voc.idx2word)) 

# Construct DDI, EHR Adj and DDI Adj data
# atc -> cid
ddi_file = 'drug-DDI.csv'
cid_atc = 'drug-atc.csv'
voc_file = 'voc_final.pkl'
data_path = 'records_final.pkl'
TOPK = 40 # topk drug-drug interaction

records =  dill.load(open(data_path, 'rb'))
cid2atc_dic = defaultdict(set)
med_voc = dill.load(open(voc_file, 'rb'))['med_voc']
med_voc_size = len(med_voc.idx2word)
med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]
atc3_atc4_dic = defaultdict(set)

for item in med_unique_word:
    atc3_atc4_dic[item[:4]].add(item)
    
with open(cid_atc, 'r') as f:
    for line in f:
        line_ls = line[:-1].split(',')
        cid = line_ls[0]
        atcs = line_ls[1:]
        for atc in atcs:
            if len(atc3_atc4_dic[atc[:4]]) != 0:
                cid2atc_dic[cid].add(atc[:4])
 
# ddi load
ddi_df = pd.read_csv(ddi_file)
ddi_topk_pd = (
    ddi_df.groupby(['Polypharmacy Side Effect', 'Side Effect Name'])
    .size()
    .reset_index(name='count')
    .sort_values('count', ascending=False)
    .tail(TOPK)
)
ddi_df = (
    ddi_df[ddi_df['Side Effect Name'].isin(ddi_topk_pd['Side Effect Name'].tolist())]
    .drop_duplicates(subset=['STITCH 1', 'STITCH 2'])
    .reset_index(drop=True)
)

# weighted ehr adj 
ehr_adj = np.zeros((med_voc_size, med_voc_size))
for patient in records:
    for adm in patient:
        med_set = adm[2]
        for i, med_i in enumerate(med_set):
            for j, med_j in enumerate(med_set):
                if j<=i:
                    continue
                ehr_adj[med_i, med_j] = 1
                ehr_adj[med_j, med_i] = 1
dill.dump(ehr_adj, open('ehr_adj_final.pkl', 'wb'))  

# ddi adj
ddi_adj = np.zeros((med_voc_size,med_voc_size))
for index, row in ddi_df.iterrows():
    # ddi
    cid1 = row['STITCH 1']
    cid2 = row['STITCH 2']
    
    # cid -> atc_level3
    for atc_i in cid2atc_dic[cid1]:
        for atc_j in cid2atc_dic[cid2]:
            
            # atc_level3 -> atc_level4
            for i in atc3_atc4_dic[atc_i]:
                for j in atc3_atc4_dic[atc_j]:
                    if med_voc.word2idx[i] != med_voc.word2idx[j]:
                        ddi_adj[med_voc.word2idx[i], med_voc.word2idx[j]] = 1
                        ddi_adj[med_voc.word2idx[j], med_voc.word2idx[i]] = 1
dill.dump(ddi_adj, open('ddi_A_final.pkl', 'wb')) 
                        
print('Complete!')
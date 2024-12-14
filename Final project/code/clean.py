import pandas as pd
import numpy as np
import copy

def clean_data(data_path, remove_feats_after_ct=True, remove_TBI_rows_with_nan=True,threshold=0.5,rm_feats=True,remove_GCS_total_mismatch=True):
    # Remove rows with missing values
    data = pd.read_csv(data_path)
    data_clean = copy.copy(data)
    # change value 92 in column to NaN for the following list of features
    list_92 = ['LocLen','SeizOccur','SeizLen','HASeverity','HAStart','VomitNbr','VomitStart','VomitLast','AMSAgitated','AMSSleep','AMSSlow','AMSRepeat','AMSOth','SFxPalpDepress','FontBulg','SFxBasHem','SFxBasOto','SFxBasPer','SFxBasRet','SFxBasRhi','HemaLoc','HemaSize','ClavFace','ClavNeck','ClavFro','ClavOcc','ClavPar','ClavTem','NeuroDMotor','NeuroDSensory','NeuroDCranial','NeuroDReflex','NeuroDOth','OSI','OSIExtremity','OSICut','OSICspine','OSIFlank','OSIAbdomen','OSIPelvis','OSIOth','Drugs','IndAge','IndAmnesia','IndAMS','IndClinSFx','IndHA','IndHema','IndLOC','IndMech','IndNeuroD','IndRqstMD','IndRqstParent','IndRqstTrauma','IndSeiz','IndVomit','IndXraySFx','IndOth','CTSed','CTSedAgitate','CTSedAge','CTSedRqst','CTSedOth','EDCT','PosCT','Finding1','Finding2','Finding3','Finding4','Finding5','Finding6','Finding7','Finding8','Finding9','Finding10','Finding11','Finding12','Finding13','Finding14','Finding20','Finding21','Finding22','Finding23']
    for i in range(len(list_92)):
        column_name = list_92[i]
        column_data = data_clean[column_name]
        #print nan indices
        indices92 = column_data[column_data == 92].index
        data_clean[column_name] = column_data.replace(92, np.nan)
        
    # change 'other' in column to NaN for the following list of features
    list_other = ['Race','EDDisposition']
    for i in range(len(list_other)):
        column_name = list_other[i]
        column_data = data_clean[column_name]
        data_clean[column_name] = column_data.replace('other', np.nan)
    # remove CTForm1
    data_clean = data_clean.drop(columns=['CTForm1'])
       
    if remove_feats_after_ct:
        # Remove features after CT
        posCT_index = data_clean.columns.get_loc('CTDone')
        data_clean = data_clean.drop(data_clean.columns[posCT_index:data_clean.shape[1]-1], axis=1)
        
    if remove_TBI_rows_with_nan:
        # Remove rows with NaN values in PosIntFinal
        data_clean = data_clean.dropna(subset=['PosIntFinal'])
    
    if rm_feats:
        # Remove features with more than threshold percent of missing values
        missing_percentage = data_clean.isnull().mean()
        missing_columns = missing_percentage[missing_percentage > threshold].index.tolist()
        data_clean = data_clean.drop(columns=missing_columns)
    
    if remove_GCS_total_mismatch:
        # remove rows where GCSTotal does not equal the sum of GCS components
        data_clean['CalculatedTotal'] = data_clean['GCSEye'] + data_clean['GCSVerbal'] + data_clean['GCSMotor']
        data_clean = data_clean[data_clean['CalculatedTotal'] == data_clean['GCSTotal']]
        data_clean = data_clean.drop(columns=['CalculatedTotal'])

    return data_clean
    
    
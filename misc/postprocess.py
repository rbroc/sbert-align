import pandas as pd 
import numpy as np
import json

def main():
    ''' Ad-hoc function to process data from study 2'''

    # Read in and get alignment data
    id_vars = ['ID', 'Visit', 'Task', 'Turn']
    df = pd.read_csv('../outputs/clean_lag-1_model-all-mpnet-base-v2.txt', sep='\t')
    df['unique_id'] = df[id_vars].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    lagged = df.shift(1)
    previous = lagged['SemanticAlignment'].tolist()
    current_to_previous = df['SemanticAlignment'].tolist()
    df = df.drop('SemanticAlignment', axis=1)
    df['alignment_current_to_1back'] = current_to_previous
    df['alignment_1back_to_2back'] = previous

    # Make sure we are only keepin stuff from the same visit
    df['lagged_Visit'] = lagged['Visit']
    df['lagged_ID'] = lagged['ID']
    df['lagged_Task'] = lagged['Task']
    df = df[(df['Visit']==df['lagged_Visit']) &
            (df['ID']==df['lagged_ID']) & 
            (df['Task']==df['lagged_Task'])]
    df['alignment_current_to_1back_type'] = df['AlignmentType']
    df['alignment_1back_to_2back_type'] = np.where(df['alignment_current_to_1back_type']=='child2caregiver',
                                                   'caregiver2child', 'child2caregiver')
    df.drop(['lagged_Visit',
             'lagged_Transcript', 
             'lagged_Task',
             'lagged_Speaker', 
             'lagged_ID', 
             'Lag',
             'index'], axis=1, inplace=True)

    # Flag problematic rows
    excl_list = json.load(open('../data/exclude.json'))
    for k in [c for c in df.columns if 'alignment_' in c]:
        df[k] = np.where(df['unique_id'].isin(excl_list['_'.join(k.split('_')[:4])]),
                         np.nan,
                         df[k])
        
    # Remove weird trial
    df = df[~((df['ID']==9611) & (df['Visit']==1) & (df['Task']=='Questions'))]

    # Save
    df.drop('AlignmentType', axis=1).to_csv('../outputs/processed_model-all-mpnet-base-v2.txt',
                                            sep='\t')


if __name__=='__main__':
    main()
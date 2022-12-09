import pandas as pd
import sentence_transformers as st
import numpy as np
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, default='all-mpnet-base-v2',
                    help='''Model name, models available at: 
                            https://www.sbert.net/docs/pretrained_models.html''')
parser.add_argument('--lag', type=int, default=1,
                    help= 'Alignment (e.g., 1 computes for previous turn''')

args = parser.parse_args()


def main(model_id, lag):
    ''' Full pipeline for alignment extraction '''

    outpath = Path('outputs')
    outpath.mkdir(exist_ok=True)
    outfile = outpath / f'model-{model_id}_lag-{lag}.tsv'

    # Read in and import
    print('*** Preprocessing data ***')
    data = pd.read_csv('data/transcripts.tsv', sep='\t')
    data = data.sort_values(by=['ChildID', 'Visit', 'Turn']).reset_index().iloc[:3000]
    data.drop([f'V{i}' for i in range(2,302)], axis=1, inplace=True)
    lagged = data.shift(1)
    for c in ['Transcript', 'Visit', 'Speaker', 'ChildID']:
        data[f'lagged_{c}'] = lagged[c]
    data.dropna(subset=['lagged_Transcript'], inplace=True)

    # Define model and similarity function
    model = st.SentenceTransformer(model_id)
    def _get_encoding(row):
        enc_0 = model.encode(row['Transcript'].tolist())
        enc_1 = model.encode(row['lagged_Transcript'].tolist())
        sim = st.util.cos_sim(enc_0, enc_1).numpy()
        return sim[np.diag_indices(sim.shape[0])].round(4)

    # Extract alignment
    print('*** Extracting alignment ***')
    grouper = data.groupby(['ChildID', 'Visit'])
    encoded = grouper.apply(_get_encoding).reset_index().explode(0)[0]
    data['SemanticAlignment'] = encoded.tolist()

    # Add metadata and remove spurious pairs
    data['AlignmentType'] = np.where(data['Speaker']=='MOT',
                                    'caregiver2child',
                                    'child2caregiver')
    data = data[(data['Visit']==data['lagged_Visit']) &
                (data['ChildID']==data['lagged_ChildID'])]
    data['Lag'] = lag
    data['ModelId'] = model_id

    # Remove redundant columns
    data = data[['ChildID', 'Visit', 'Turn',
                 'Lag', 'ModelId',
                 'SemanticAlignment',
                 'AlignmentType']]
    data.to_csv(str(outfile), sep='\t', index=False)


if __name__=='__main__':
    args = parser.parse_args()
    main(args.model, args.lag)

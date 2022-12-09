import pandas as pd
import sentence_transformers as st
import numpy as np
import swifter

# Read in and import
df = pd.read_csv('data/transcripts.txt', sep='\t')
df = df.sort_values(by=['ChildID', 'Visit', 'Turn']).reset_index()
df.drop([f'V{i}' for i in range(2,302)], axis=1, inplace=True)
lagged = df.shift(1)
for c in ['Transcript', 'Visit', 'Speaker', 'ChildID']:
    df[f'lagged_{c}'] = lagged[c]
df.dropna(subset=['lagged_Transcript'], inplace=True)

# Compute alignment
model = st.SentenceTransformer('all-MiniLM-L6-v2')

def _get_encoding(row):
    enc_0 = model.encode(row['Transcript'])
    enc_1 = model.encode(row['lagged_Transcript'])
    sim = round(float(st.util.cos_sim(enc_0, enc_1)),4)
    return sim

# Add metadata
df['SemanticAlignment'] = df.swifter.apply(_get_encoding, axis=1)
df['AlignmentType'] = np.where(df['Speaker']=='MOT',
                               'caregiver2child',
                               'child2caregiver')

# Keep rows where lagged and reference transcript have same ID and Visit
df = df[(df['Visit']==df['lagged_Visit']) &
        (df['ChildID']==df['lagged_ChildID'])]

# Remove non-consecutive turns and visits

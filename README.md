# sbert-align
Compute parent-child alignment using SentenceBERT

### Usage
1. Create a virtual environment (not necessary).

You can do so by typing:

``` 
python3 -m venv PATH_TO_ENV
source PATH_TO_ENV/bin/activate
```
Replace `PATH_TO_ENV` with path for virtual environment

2. Install requirements
```pip install -r requirements.txt```

3. Run the `align.py` script.

`python3 align.py --lag 1 --model all-mpnet-base-v2`.

Arguments are customizable.

Note that the script will be looking for a `transcripts.txt` or `surrogates.txt` file in the `data` folder, and outputs will be saved in an `outputs` folder.

4. Deactivate once you're done, by running ```deactivate```.

### Output columns
- Turn metadata: (`ChildID|ID`, `Visit`, `Turn`)
- `Lag`: 1 if alignment is computed with previous turn, 2 if two turns back. Note that even numbers compute alignment with previous turns from same speaker;
- `ModelId`: Which SentenceBERT checkpoint we are using, see https://www.sbert.net/docs/pretrained_models.html for available models; 
- `SemanticAlignment`: cosine similarity between sequence encodings;
- `AlignmentType`: 'child2caregiver' or 'caregiver2child'

### Potential expansion:
- Make synthetic raw data for better reproducibility
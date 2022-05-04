
# Precise Spelling Correction

Targeted spell correction using edit distance and an initial vocabulary of correct spellings.

This library targets only a specific number of words, generating possible misspellings, ensuring they don't appear in an existing vocabulary, and looking for those misspellings in text documents.

## Usage

See `tests` directory for additional help.

### Prerequisites

* `search_terms.txt`: text file of terms to try to find misspellings of; one word per line
* `vocabulary.tsv`: vocabulary file to determine what proper spellings are; tab-separated file of `word\tfrequency\n`
  * word: text string containing no spaces
  * frequency: number of times this word appears in corpus
* A directory of text files (`input_directory`)


### Example

From code:

```python
import pathlib
from precise_spell_correction.correct_text import  correct_text_in_directory

data_path = pathlib.Path(r'/path/to/data')
text_path = pathlib.Path(r'/path/to/text_files')
out_path = pathlib.Path(r'/path/to/output')
correct_text_in_directory(
        search_terms=['acknowledgment', 'accommodate'],
        search_term_path=data_path / 'search_terms.txt',  # one word per line
        input_directory=text_path,
        output_directory=out_path,
        vocab_file=data_path / 'vocabulary.tsv',  # required
        min_freq=1,  # minimum frequency to count something as a word; 
                     # min_freq=1 expects entire vocab to have only actual words
        dbfile=data_path / 'spelling.db',  # will be created
        edit_distance=1
    )
```

From command line:

1. If installed (`pip install .` from inside `precise_spelling_corrector` directory):

```shell
run-precise-spelling-corrector 
  --search-terms acknowledgment accommodate
  --search-term-path /path/search_terms.txt
  --input-directory /path/indir
  --output-directory /path/outdir
  --dbfile /spelling.db
  --min-freq 1
  --edit-distance 1
  --vocab-file /path/vocabulary.tsv
```

2. Otherwise (not installed):

```shell
export/set PYTHONPATH=src  # or $env:PYTHONPATH=src
cd precise_spell_correction
python correct_text.py 
  --search-terms acknowledgment accommodate
  --search-term-path /path/search_terms.txt
  --input-directory /path/indir
  --output-directory /path/outdir
  --dbfile /spelling.db
  --min-freq 1
  --edit-distance 1
  --vocab-file /path/vocabulary.tsv
```

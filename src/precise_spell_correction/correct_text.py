import difflib
import os
import pathlib
import sqlite3
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Pattern

from regexify import PatternTrie
import string
import regex as re
from loguru import logger

LETTERS = ' ' + string.ascii_letters + string.digits + string.punctuation


TRANSFORMATION = None


class Vocab:

    def __init__(self):
        self.data = {}
        self.longest_word = ''
        self._punct_table = str.maketrans({key: ' ' for key in string.punctuation})

    def _exclude_common_misspell(self, term):
        if term[:3] == 'the':
            return False
        return True

    def get_closest_match(self, word):
        lst = difflib.get_close_matches(word.strip(), self.data.keys(), n=100, cutoff=0.70)
        lst = [el for el in lst if self._exclude_common_misspell(el)]
        if not lst:
            return None
        return max((self.data[el], el) for el in lst)[1]

    @property
    def keys(self):
        return set(self.data.keys())

    @property
    def longest_word_length(self):
        return len(self.longest_word) if self.longest_word else 0

    @property
    def word_count(self):
        return len(self.data)

    @classmethod
    def load_from_file(cls, filename, min_freq=2):
        vocab = cls()
        it = vocab._file_to_iter(filename)
        vocab.add_words_from_iter(it, min_freq)
        return vocab

    @classmethod
    def load_from_list(cls, lst, min_freq=2):
        vocab = cls()
        if not isinstance(lst[0], tuple):
            lst = ((w, 1) for w in lst)
        vocab.add_words_from_iter(lst, min_freq)
        return vocab

    def _clean_terms(self, term):
        if not term.strip():
            return
        row = term.strip().split('\t')
        if len(row) >= 2:
            yield row[0].strip(), int(row[1])
        else:
            yield row[0].strip(), 1

    def add_word(self, term, freq):
        if len(term) > len(self.longest_word):
            self.longest_word = term
        self.data[term] = freq

    def add_words_from_iter(self, it, min_freq=10):
        for term, freq in it:
            if freq >= min_freq:
                self.add_word(term, freq)

    def _file_to_iter(self, input_file, encoding='utf8'):
        if not input_file:
            return
        with open(input_file, encoding=encoding) as fh:
            for line in fh:
                for term, freq in self._clean_terms(line):
                    yield term, freq

    def __contains__(self, item):
        if item in self.data:
            return True
        no_punct = item.translate(self._punct_table).strip()
        if no_punct in self.data:
            return True
        for w in no_punct.split():
            if w not in self.data:
                return False
        return True

    def __getitem__(self, item):
        return self.data.get(item, 0)

    def check_if_should_check(self, word):
        if word in self:
            return False
        if len(word) == 1 and word in string.punctuation:
            return False
        if len(word) > self.longest_word_length + 3:  # magic number to allow removal of up to 2 letters.
            return False
        try:  # check if it is a number (int, float, etc)
            float(word)
            return False
        except ValueError:
            pass
        return True


def ensure_unicode(s, encoding='utf-8'):
    if isinstance(s, bytes):
        return s.decode(encoding)
    return s


class Transformations:
    """Keep track of transformations"""

    def __init__(self):
        self._transform = Counter()
        self._transform_context = defaultdict(list)
        self._failed = Counter()
        self._failed_context = defaultdict(list)

    def transform(self, src, dest, context=None):
        self._transform[(src, dest)] += 1
        if context:
            self._transform_context[src].append(context)

    def failed_to_transform(self, term, context):
        self._failed[term] += 1
        if context:
            self._failed_context[term].append(context)

    def to_file(self, d: pathlib.Path, encoding='utf8'):
        with open(d / 'transform_freq.txt', 'w', encoding=encoding) as out:
            for (src, dest), cnt in self._transform.most_common():
                out.write(f'{src}\t{dest}\t{cnt}\n')
        with open(d / 'no_transform.txt', 'w', encoding=encoding) as out:
            for term, cnt in self._failed.most_common():
                out.write(f'{term}\t{cnt}\n')
        with open(d / 'transform_context.txt', 'w', encoding=encoding) as out:
            for term, context in self._transform_context.items():
                out.write(f'{term}\t{context}\n')
        with open(d / 'no_transform_context.txt', 'w', encoding=encoding) as out:
            for term, context in self._failed_context.items():
                out.write(f'{term}\t{context}\n')


class Data:

    def __init__(self, dbfile):
        self.exists = os.path.isfile(dbfile)
        self.conn = sqlite3.connect(dbfile)
        self.cur = self.conn.cursor()
        if not self.exists:
            self.commit('create table lookup (variation text, target text, edit_distance integer)')
            self.commit('create table metadata (key text, value text)')
            self._initialize()

    def _initialize(self):
        self.commit('insert into metadata (key, value) values (?, ?)', 'pattern1', '')
        self.commit('insert into metadata (key, value) values (?, ?)', 'pattern2', '')
        self.save()

    def delete(self):
        self.commit('delete from lookup')
        self.commit('delete from metadata')
        self._initialize()

    def commit(self, query, *args):
        self.cur.execute(query, args)
        # self.conn.commit()

    def save(self):
        self.conn.commit()

    @lru_cache(maxsize=256)
    def execute_fetchone(self, query, *args):
        self.save()
        val = self.cur.execute(query, args).fetchone()
        return val[0] if val else None

    def get(self, metadata_key):
        return self.execute_fetchone(f'select value from metadata where key = ?', metadata_key)

    def add(self, metadatakey, metadatavalue):
        self.commit(f"insert into metadata (key, value) values (?, ?)", metadatakey, metadatavalue)

    def set(self, metadatakey, metadatavalue):
        self.commit(f"update metadata set value = ? where key = ?", metadatavalue, metadatakey)

    def __getitem__(self, item):
        return self.execute_fetchone(f'select target from lookup where variation = ?', item)

    def __setitem__(self, key, value):
        self.set_edit_distance(key, value, edit_distance=1)

    def set_edit_distance(self, key, value, edit_distance=1):
        self.commit(f"insert into lookup (variation, target, edit_distance) values (?, ?, ?)", key, value,
                    edit_distance)

    def __contains__(self, item):
        return self.execute_fetchone(f'select target from lookup where variation = ?', item) is not None

    def __len__(self):
        return self.execute_fetchone(f'select count(*) from lookup')

    def keys(self):
        self.save()
        for r in self.cur.execute('select distinct target from lookup'):
            yield r[0]

    def values(self, edit_distance=1):
        self.save()
        for r in self.cur.execute(f'select variation from lookup where edit_distance = ?', (edit_distance,)):
            yield r[0]

    def contains_term(self, term):
        return self.execute_fetchone(f"select variation from lookup where target = ? limit 1", term) is not None


class SpellCorrector:

    def __init__(self, vocab: Vocab, dbfile='spell_corrector_terms'):
        self.data = Data(dbfile)
        self.vocab = vocab
        self._pattern = re.compile(self.data.get('pattern1'), re.I | re.ENHANCEMATCH)  # no word boundary, enhanced
        self._pattern2 = re.compile(self.data.get('pattern2'), re.I)  # strict, word boundary
        self._terms = []

    @property
    def pattern(self):
        return self._pattern

    @property
    def pattern_strict(self):
        return self._pattern2

    def __bool__(self):
        return len(self.data) > 0

    def __contains__(self, item):
        return item in self.data

    def __getitem__(self, item):
        return self.data[item]

    def _larger_word_in_vocab(self, pre, capture, post):
        context_word = f'{pre}{capture}{post}'.lower()
        if capture.lower() == context_word.strip():
            return False
        for word in context_word.split():
            if word not in self.vocab:
                return False
        return True

    def _splititer(self, pat: Pattern, text, context):
        prev = 0
        for m in pat.finditer(text):
            skipped_text = text[prev:m.start()]
            capture = text[m.start():m.end()]
            # are we in the middle of a word?
            pretext = text[m.start() - context:m.start()]
            posttext = text[m.end():m.end() + context]
            pre = re.split(r'\W+', pretext)[-1]
            post = re.split(r'\W+', posttext)[0]
            if self._larger_word_in_vocab(pre, capture, post):
                continue  # don't update prev, this will all be `skipped_text` next iteration
            if len(pre) > 2 and pre in self.vocab:
                skipped_text += ' '  # this should be a separate word
            context_text = ' '.join(text[m.start() - context: m.end() + context].split())
            prev = m.end()
            yield skipped_text, capture, context_text
        yield text[prev:], None, None

    def splititer(self, text, context=20):
        yield from self._splititer(self._pattern, text, context)

    def splititer_strict(self, text, context=20):
        yield from self._splititer(self._pattern2, text, context)

    def splititer_word(self, text, context=20):
        yield from self._splititer(re.compile(r'(\w+)', re.I), text, context)

    def _is_first_last_letter(self, word):
        if word[0] in string.ascii_lowercase and word[-1] in string.ascii_lowercase:
            return True
        return False

    def update_pattern(self):
        self._pattern = re.compile(rf'({PatternTrie(*self._terms).pattern}){{e<=2:\S}}', re.I | re.ENHANCEMATCH)
        self.data.set('pattern1', self._pattern.pattern)
        self._pattern2 = re.compile(
            rf'\b({PatternTrie(*(x for x in self.data.values() if self._is_first_last_letter(x))).pattern})'
            rf'{{e<=2:[A-Za-z\s]}}\b',
            re.I
        )
        self.data.set('pattern2', self._pattern2.pattern)
        self.data.save()

    def add_spelling_variant_pattern(self, search_terms, *, edit_distance=2, reload=False):
        logger.info(f'Edit distance: {edit_distance}')
        keys = set(self.data.keys())
        if reload:
            self.data.delete()
        for term in search_terms:
            logger.info(f'Loading {term}')
            term = term.lower()
            self._terms.append(term)
            self.vocab.add_word(term, 10000)
            if term in keys:
                logger.info(f'Already loaded: {term}')
                continue
            # get edit distance 1...
            for r in self.edit_distance_1(term):
                self.data[r] = term
                if edit_distance == 2:
                    for r2 in self.edit_distance_alt(r):
                        self.data.set_edit_distance(r2, term, edit_distance=2)
        self.update_pattern()

    def add_from_file(self, filename, *, edit_distance=2):
        search_terms = []
        with open(filename, encoding='utf8') as fh:
            for line in fh:
                search_terms.append(line.strip().lower())
        self.add_spelling_variant_pattern(search_terms, edit_distance=edit_distance)

    def edit_distance_1(self, word, check_vocab=True):
        """ Compute all strings that are one edit away from `word` using only
            the letters in the corpus
            Args:
                word (str): The word for which to calculate the edit distance
            Returns:
                set: The set of strings that are edit distance one from the \
                provided word """
        word = ensure_unicode(word).lower()
        if check_vocab and self.vocab.check_if_should_check(word):
            return {word}
        letters = LETTERS
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        # transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(x for x in deletes + replaces + inserts if
                   x[0] in string.ascii_lowercase and x[-1] in string.ascii_lowercase and x not in self.vocab)

    def edit_distance_2(self, word):
        """ Compute all strings that are two edits away from `word` using only
            the letters in the corpus
            Args:
                word (str): The word for which to calculate the edit distance
            Returns:
                set: The set of strings that are edit distance two from the \
                provided word """
        word = ensure_unicode(word).lower()
        return [
            e2 for e1 in self.edit_distance_1(word) for e2 in self.edit_distance_1(e1)
        ]

    def edit_distance_alt_word(self, word):
        word = word.lower()
        if not self.vocab.check_if_should_check(word):
            return
        for e2 in self.edit_distance_1(word, check_vocab=False):
            if self.vocab.check_if_should_check(e2):
                yield e2

    def edit_distance_alt(self, words):
        """ Compute all strings that are 1 edits away from all the words using
            only the letters in the corpus
            Args:
                words (list): The words for which to calculate the edit distance
            Returns:
                set: The set of strings that are edit distance two from the \
                provided words """
        words = (ensure_unicode(w).lower() for w in words)
        words = (w for w in words if self.vocab.check_if_should_check(w))
        for e1 in words:
            yield from self.edit_distance_alt_word(e1)


def retain_word_shape(old_word: str, new_word: str):
    if old_word.isupper():
        return new_word.upper()
    elif old_word[0].isupper():
        return new_word.capitalize()
    return new_word


def spell_correct_words(text_iter, sc: SpellCorrector, transform: Transformations, *, get_closest_match=False):
    for nonword, word, context in text_iter:
        yield nonword
        if not word:
            break
        lword = word.lower()
        if lword in sc.vocab:  # word already known
            yield word
        elif lword in sc:
            new_word = sc[lword]
            transform.transform(lword, new_word, context)
            logger.info(f'Changing {lword} to {new_word} ({context})')
            yield retain_word_shape(word, new_word)
        elif get_closest_match:
            new_word = sc.vocab.get_closest_match(lword)
            if new_word is None:
                transform.failed_to_transform(lword, context)
                logger.warning(f'Failed to find term corresponding to {lword}, even though it was matched ({context})')
                yield word
            else:
                logger.info(f'Used lookup to change {lword} to {new_word} ({context})')
                if lword[0] == ' ' or lword[-1] == ' ':
                    new_word = f' {new_word} '
                yield retain_word_shape(word, new_word)
        else:
            transform.failed_to_transform(lword, context)
            logger.warning(f'Failed to find term corresponding to {lword}, even though it was matched ({context})')
            yield word


def iteratively_correct_text(source, dest, sc: SpellCorrector, transform: Transformations, *, use_regex=False):
    dest.mkdir(exist_ok=True)
    for source_path in (p for p in source.rglob('*') if p.is_file()):
        dest_path = (dest / source_path.relative_to(source)).resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(source_path, encoding='utf8') as fh, open(dest_path, 'w', encoding='utf8') as out:
            text = ''.join([i if ord(i) < 128 else ' ' for i in fh.read()])  # strip non-ascii
            if use_regex:  # by word
                for segment in spell_correct_words(sc.splititer_word(text), sc, transform):
                    out.write(segment)
            else:
                logger.info('# Stage 1')
                text = ''.join([x for x in spell_correct_words(sc.splititer(text), sc, transform)])
                logger.info('# Stage 2')
                for segment in spell_correct_words(sc.splititer_strict(text), sc, transform, get_closest_match=True):
                    out.write(segment)


def correct_text_in_directory(search_terms, search_term_path, input_directory, output_directory, vocab_file, 
                              *, min_freq=1, dbfile='spell_corrector_terms', edit_distance=2, **kwargs):
    vocab = Vocab.load_from_file(vocab_file, min_freq=min_freq)
    sc = SpellCorrector(vocab, dbfile=dbfile)
    sc.add_from_file(search_term_path, edit_distance=edit_distance)
    sc.add_spelling_variant_pattern(search_terms, edit_distance=edit_distance)
    if not sc:
        raise ValueError('No search terms provided.')
    source = pathlib.Path(input_directory)
    dest = pathlib.Path(output_directory)
    transform = Transformations()
    iteratively_correct_text(source, dest, sc, transform)
    transform.to_file(dest)


def get_transform(create_new=False):
    """Get a transformation; stored globally to allow easy retrieval of results"""
    global TRANSFORMATION
    if create_new or TRANSFORMATION is None:
        TRANSFORMATION = Transformations()
    return TRANSFORMATION


def correct_text(text, sc, transform=None):
    """Spell correct any text given a `SpellCorrector` and a `Transformations`"""
    if not transform:
        transform = get_transform()
    text = ''.join([x for x in spell_correct_words(sc.splititer(text), sc, transform)])
    return ''.join(spell_correct_words(sc.splititer_strict(text), sc, transform, get_closest_match=True))


def correct_text_dataframe(df, colname, term_path, vocab_path, *, min_freq=3, edit_distance=1, 
                           transform=None):
    vocab = Vocab.load_from_file(vocab_path, min_freq=min_freq)
    sc = SpellCorrector(vocab, dbfile='search_term_file')
    sc.add_from_file(term_path, edit_distance=edit_distance)
    if not transform:
        transform = get_transform()
    return df[colname].apply(
        lambda text: correct_text(text, sc, transform)
    )


def correct_text_cmd():
    import argparse
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@!')
    parser.add_argument('--search-terms', nargs='+', dest='search_terms', default=list(),
                        help='Terms to search for')
    parser.add_argument('--search-term-file', dest='search_term_file', default=None,
                        help='Terms to search for in a file, one word per line')
    parser.add_argument('--vocab-file', dest='vocab_file', required=True,
                        help='File containing entire vocab, one word per line;'
                             r' word\tfreq is acceptable for use with min-freq option')
    parser.add_argument('--min-freq', dest='min_freq', type=int, default=3,
                        help='Minimum word frequency to include from vocab')
    parser.add_argument('--edit-distance', dest='edit_distance', type=int, default=1,
                        help='Edit distance measure, defaults to 2')
    parser.add_argument('--dbfile', default='spell_corrector_terms',
                        help='Path to sqlite database file.')
    parser.add_argument('--input-directory', dest='input_directory', required=True,
                        help='Directory, all containing files will be spell-corrected')
    parser.add_argument('--output-directory', dest='output_directory', required=True,
                        help='Directory, will create matching input-directory, but spell corrected;'
                             ' logs will also go here')
    args = parser.parse_args()

    logger.add(pathlib.Path(args.output_directory) / 'correct_text.log')
    correct_text_in_directory(args.search_terms, args.search_term_file, 
                              args.input_directory, args.output_directory,
                              args.vocab_file, 
                              min_freq=args.min_freq, dbfile=args.dbfile, edit_distance=args.edit_distance
                              )


if __name__ == '__main__':
    correct_text_cmd()

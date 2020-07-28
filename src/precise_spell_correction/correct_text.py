import pathlib
from collections import Counter

from regexify import PatternTrie
import string
import regex as re
from loguru import logger

LETTERS = ' ' + string.ascii_letters + string.digits + string.punctuation


class Vocab:

    def __init__(self):
        self.data = {}
        self.longest_word = ''

    @property
    def longest_word_length(self):
        return len(self.longest_word) if self.longest_word else 0

    @property
    def word_count(self):
        return len(self.data)

    @classmethod
    def load_from_file(cls, filename):
        vocab = cls()
        it = vocab._file_to_iter(filename)
        vocab.add_words_from_iter(it)
        return vocab

    @classmethod
    def load_from_list(cls, lst):
        vocab = cls()
        if not isinstance(lst[0], tuple):
            lst = ((w, 1) for w in lst)
        vocab.add_words_from_iter(lst)
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

    def add_words_from_iter(self, it):
        for term, freq in it:
            self.add_word(term, freq)

    def _file_to_iter(self, input_file, encoding='utf8'):
        if not input_file:
            return
        with open(input_file, encoding=encoding) as fh:
            for line in fh:
                for term, freq in self._clean_terms(line):
                    yield term, freq

    def __contains__(self, item):
        return item in self.data

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
        self._failed = Counter()

    def transform(self, src, dest):
        self._transform[(src, dest)] += 1

    def failed_to_transform(self, term):
        self._failed[term] += 1

    def to_file(self, d: pathlib.Path):
        with open(d / 'transform_freq.txt', 'w') as out:
            for (src, dest), cnt in self._transform.most_common():
                out.write(f'{src}\t{dest}\t{cnt}\n')
        with open(d / 'no_transform.txt', 'w') as out:
            for term, cnt in self._failed.most_common():
                out.write(f'{term}\t{cnt}\n')


class SpellCorrector:

    def __init__(self, vocab, ):
        self.data = {}
        self.vocab = vocab
        self._pattern = None
        self._terms = []

    @property
    def pattern(self):
        return self._pattern

    def __bool__(self):
        return len(self.data) > 0

    def __contains__(self, item):
        return item in self.data

    def __getitem__(self, item):
        return self.data[item]

    def splititer(self, text):
        return self._pattern.splititer(text)

    def update_pattern(self):
        self._pattern = re.compile(rf'({PatternTrie(*self._terms).pattern}){{e<=2:\S}}', re.I)

    def add_spelling_variant_pattern(self, search_terms, *, edit_distance=2):
        for term in search_terms:
            term = term.lower()
            self._terms.append(term)
            self.vocab.add_word(term, 1)
            # get edit distance 1...
            res = [x for x in self.edit_distance_1(term)]
            # if still not found, use the edit distance 1 to calc edit distance 2
            if edit_distance == 2:
                res = [x for x in self.edit_distance_alt(res)]
            for r in res:
                self.data[r] = term
        self.update_pattern()

    def add_from_file(self, filename, *, edit_distance=2):
        search_terms = []
        with open(filename, encoding='utf8') as fh:
            for line in fh:
                search_terms.append(line.strip().lower())
        self.add_spelling_variant_pattern(search_terms, edit_distance=edit_distance)

    def edit_distance_1(self, word):
        """ Compute all strings that are one edit away from `word` using only
            the letters in the corpus
            Args:
                word (str): The word for which to calculate the edit distance
            Returns:
                set: The set of strings that are edit distance one from the \
                provided word """
        word = ensure_unicode(word).lower()
        if self.vocab.check_if_should_check(word):
            return {word}
        letters = LETTERS
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

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
        return [e2 for e1 in words for e2 in self.edit_distance_1(e1)]


def retain_word_shape(old_word: str, new_word: str):
    if old_word.isupper():
        return new_word.upper()
    elif old_word[0].isupper():
        return new_word.capitalize()
    return new_word


def spell_correct_words(text, sc: SpellCorrector, transform: Transformations, *, use_regex=True):
    text = ''.join([i if ord(i) < 128 else ' ' for i in text])  # strip non-ascii
    if use_regex:
        it = sc.splititer(text)
    else:  # by word
        it = re.splititer(r'(\w+)', text, flags=re.I)
    for nonword in it:
        yield nonword
        try:
            word = next(it)
        except StopIteration:
            break
        lword = word.lower()
        if lword in sc.vocab:  # word already known
            yield word
        elif lword in sc:
            new_word = sc[lword]
            transform.transform(lword, new_word)
            logger.info(f'Changing {lword} to {new_word}')
            yield retain_word_shape(word, new_word)
        else:
            transform.failed_to_transform(lword)
            logger.warning(f'Failed to find term corresponding to {lword}, even though it was matched.')
            yield word


def iteratively_correct_text(source, dest, sc: SpellCorrector, transform: Transformations):
    dest.mkdir(exist_ok=True)
    for source_path in (p for p in source.rglob('*') if p.is_file()):
        dest_path = (dest / source_path.relative_to(source)).resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(source_path, encoding='utf8') as fh, open(dest_path, 'w', encoding='utf8') as out:
            for segment in spell_correct_words(fh.read(), sc, transform):
                out.write(segment)


def correct_text(search_terms, search_term_path, input_directory, output_directory, vocab_file, min_freq=1, **kwargs):
    vocab = Vocab.load_from_file(vocab_file)
    sc = SpellCorrector(vocab)
    sc.add_from_file(search_term_path)
    sc.add_spelling_variant_pattern(search_terms)
    if not sc:
        raise ValueError('No search terms provided.')
    source = pathlib.Path(input_directory)
    dest = pathlib.Path(output_directory)
    transform = Transformations()
    iteratively_correct_text(source, dest, sc, transform)
    transform.to_file(dest)


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
    parser.add_argument('--min-freq', dest='min_freq', type=int, default=1,
                        help='Minimum word frequency to include from vocab')
    parser.add_argument('--input-directory', dest='input_directory', required=True,
                        help='Directory, all containing files will be spell-corrected')
    parser.add_argument('--output-directory', dest='output_directory', required=True,
                        help='Directory, will create matching input-directory, but spell corrected;'
                             ' logs will also go here')
    args = parser.parse_args()

    logger.add(pathlib.Path(args.output_directory) / 'correct_text.log')

    correct_text(args.search_terms, args.search_term_file, args.input_directory, args.output_directory,
                 args.vocab_file, args.min_freq
                 )


if __name__ == '__main__':
    correct_text_cmd()

from pathlib import Path

import click
from loguru import logger

from precise_spell_correction import Vocab, SpellCorrector, Transformations
from precise_spell_correction.correct_text import spell_correct_words
from precise_spell_correction.readers import CsvIO, JsonlIO, TextIO, DirIO


@click.command()
@click.argument('text_file', type=click.Path(path_type=Path, exists=True))
@click.argument('vocab_file', type=click.Path(path_type=Path, exists=True))
@click.argument('search_term_path', type=click.Path(path_type=Path, exists=True))
@click.option('--edit-distance', type=int, default=1,
              help='Set edit distance (suggested 1 or 2)')
@click.option('--header', type=str, default=None)
def _main(text_file: Path, vocab_file: Path, search_term_path: Path, edit_distance=1, header=None):
    main(text_file, vocab_file, search_term_path, edit_distance=edit_distance, header=header)


def main(text_file: Path, vocab_file: Path, search_term_path: Path, edit_distance=1, header=None):
    sc, transform = get_spell_corrector(vocab_file, search_term_path, edit_distance=edit_distance)
    with get_text_iter(text_file, header=header) as text_io:
        for text in text_io:
            new_text = correct_text(text, sc, transform)
            text_io.write(new_text)


def get_text_iter(text_file, header=None, encoding='utf8'):
    if text_file.suffix == '.csv':
        return CsvIO(text_file, header, encoding=encoding)
    elif text_file.suffix == '.jsonl':
        return JsonlIO(text_file, header, encoding=encoding)
    elif text_file.is_file:
        return TextIO(text_file, header, encoding=encoding)
    elif text_file.is_dir():
        return DirIO(text_file, header, encoding=encoding)
    raise ValueError(f'Unrecognized kind of text file: {text_file}. Does it exist?')


def get_spell_corrector(vocab_file: Path, search_term_path: Path, edit_distance=1):
    vocab = Vocab.load_from_file(vocab_file, min_freq=1)
    sc = SpellCorrector(vocab)
    sc.add_from_file(search_term_path, edit_distance=edit_distance)
    transform = Transformations()
    return sc, transform


def _correct_text(text, sc, transform):
    text = ''.join([x for x in spell_correct_words(sc.splititer(text), sc, transform)])
    text = ''.join([x for x in spell_correct_words(sc.splititer_strict(text), sc, transform, get_closest_match=True)])
    return text


def correct_text(text, sc, transform):
    text = _correct_text(text, sc, transform)
    return ''.join([x for x in spell_correct_words(sc.splititer(text), sc, transform)])


if __name__ == '__main__':
    _main()

import pathlib
import shutil

import pytest
from hypothesis import given
import hypothesis.strategies as st
from string import ascii_letters, ascii_lowercase

from precise_spell_correction.correct_text import iteratively_correct_text, SpellCorrector, Vocab


def inpath():
    return pathlib.Path(__file__).parent / 'test_inpath'


def outpath():
    return pathlib.Path(__file__).parent / 'test_outpath'


def create_random_path(dirs, file):
    return pathlib.Path(inpath(), *dirs, file)


@pytest.fixture()
def spell_corrector():
    sc = SpellCorrector(Vocab.load_from_list(['asdflkj']))
    sc.add_spelling_variant_pattern('alskdfj')
    return sc


@given(paths=st.lists(st.builds(create_random_path,
                                st.lists(
                                    st.text(alphabet=st.sampled_from(ascii_lowercase + '0123456789'), min_size=4,
                                            max_size=15),
                                    min_size=2, max_size=10),
                                st.text(alphabet=st.sampled_from(ascii_letters + '0123456789' + '.'), min_size=1,
                                        max_size=10),
                                ), min_size=1, max_size=10))
def test_random_directory_matched(paths, spell_corrector):
    inpath().mkdir(exist_ok=True)
    outpath().mkdir(exist_ok=True)
    for path in paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return
        with open(path, 'w', encoding='utf8') as out:
            out.write('test')
    iteratively_correct_text(inpath(), outpath(), spell_corrector)
    infiles = set(x.relative_to(inpath()) for x in inpath().rglob('*') if x.is_file())
    outfiles = set(x.relative_to(outpath()) for x in outpath().rglob('*') if x.is_file())
    assert infiles == outfiles
    shutil.rmtree(inpath())
    shutil.rmtree(outpath())

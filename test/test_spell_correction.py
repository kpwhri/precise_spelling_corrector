from precise_spell_correction.correct_text import SpellCorrector, Vocab, spell_correct_words


def test_spell_correction():
    vocab = Vocab.load_from_list(['today', 'is', 'a', 'nice', 'day'])
    sc = SpellCorrector(vocab)
    sc.add_spelling_variant_pattern(['nice'])
    sentence = 'Today is a niec day.'
    expected = 'Today is a nice day.'
    assert ''.join(spell_correct_words(sentence, sc)) == expected

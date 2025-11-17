import pytest

from ai_security.rules_chatter_detector import (
    simple_raw_chatter_detector,
    simple_normalized_chatter_detector
)


@pytest.mark.parametrize(
    'transcript,expected',
    [
        (
            '''Hi it's me. He knows the whole story. Bourne's just the tip of the
            iceberg. Have you heard of an "Operation Blackbriar"? I'm going to
            get my head around this and type it up. I'll see you first thing.
            Okay.''',
            'harmless'
        ),
        (
            'operation blackbriar',
            'blackbriar'
        )
    ]
)
def test_simple_raw_chatter_detector(transcript, expected):
    res = simple_raw_chatter_detector(transcript)

    assert res == expected


@pytest.mark.parametrize(
    'transcript,expected',
    [
        (
            '''Hi it's me. He knows the whole story. Bourne's just the tip of the
            iceberg. Have you heard of an "Operation Blackbriar"? I'm going to
            get my head around this and type it up. I'll see you first thing.
            Okay.''',
            'blackbriar'
        ),
        (
            'operation blackbriar',
            'blackbriar'
        ),
        (
            'In the time of chimpanzees, I was a monkey',
            'harmless'
        )
    ]
)
def test_simple_normalized_chatter_detector(transcript, expected):
    res = simple_normalized_chatter_detector(transcript)

    assert res == expected

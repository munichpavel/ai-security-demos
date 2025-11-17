from enum import Enum


def simple_raw_chatter_detector_blackbriar(transcript: str) -> str:
    if 'operation blackbriar' in transcript or 'blackbriar' in transcript:
        print("Key-word 'blackbriar' found in transcript")
        chatter_category = 'blackbriar'
    else:
        print("Transcript harmless")
        chatter_category = 'harmless'

    return chatter_category




def simple_raw_chatter_detector(transcript: str) -> str:
    if 'operation blackbriar' in transcript or 'blackbriar' in transcript:
        print("Key-word 'blackbriar' found in transcript")
        chatter_category = 'blackbriar'
    elif 'operation treadstone' in transcript or 'treadstone' in transcript:
        print("Key-word 'treadstone' found in transcript")
        chatter_category = 'treadstone'
    elif 'ultra' in transcript:
        print("Key-word 'ultra' found in transcript")
        chatter_category = 'ultra'
    else:
        print("Transcript harmless")
        chatter_category = 'harmless'

    return chatter_category


def simple_normalized_blacbriar_chatter_detector(transcript: str) -> str:
    normalized_transcript = transcript.lower()
    chatter_category = simple_raw_chatter_detector_blackbriar(transcript=normalized_transcript)

    return chatter_category


def simple_normalized_chatter_detector(transcript: str) -> str:
    normalized_transcript = transcript.lower()
    chatter_category = simple_raw_chatter_detector(transcript=normalized_transcript)

    return chatter_category

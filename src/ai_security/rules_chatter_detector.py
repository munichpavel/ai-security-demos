def simple_raw_chatter_detector_blackbriar(transcript: str) -> str:
    # Set the default chatter category to 'harmless'
    chatter_category = 'harmless'

    chatter_keywords = ['operation blackbriar', 'blackbriar']

    # Go one by one through chatter keywords
    for a_keyword in chatter_keywords:
        # If one of the keywords is found inside the transcript, then ...
        if a_keyword in transcript:
            # set the chatter category to 'blackbriar'.
            chatter_category = 'blackbriar'
            print(f'Keyword {a_keyword} found in chatter')

    return chatter_category


def simple_normalized_blackbriar_chatter_detector(transcript: str) -> str:
    normalized_transcript = transcript.lower()
    chatter_category = simple_raw_chatter_detector_blackbriar(transcript=normalized_transcript)

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


def simple_normalized_chatter_detector(transcript: str) -> str:
    normalized_transcript = transcript.lower()
    chatter_category = simple_raw_chatter_detector(transcript=normalized_transcript)

    return chatter_category

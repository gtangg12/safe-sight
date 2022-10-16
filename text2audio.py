import os
import ffmpeg
from pydub import AudioSegment
import math
import random
import shutil

from google.cloud import texttospeech


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "authentication.json"


def synthesize_text_chinese(text: str, output_string: str):
    '''
    Given a text string and a output file name, this function will 
    write the audio of the text to the output file in Chinese
    '''
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    voice = texttospeech.VoiceSelectionParams(
        language_code="yue-HK",
        name="yue-HK-Standard-A",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )

    audio_config = texttospeech.AudioConfig(
        pitch = 3.2,
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    # The response's audio_content is binary.
    with open(output_string, "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

def synthesize_text_english(text: str, output_file_name: str):
    '''
    Given a text string and a output file name, this function will 
    write the audio of the text to the output file in English
    '''
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="US-wavenet-F",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )

    audio_config = texttospeech.AudioConfig(
        pitch = 4.2,
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    # The response's audio_content is binary.
    with open(output_file_name, "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

def gen_silence():
    '''
    Generates silence to be used between text snippets
    '''
    if random.uniform(0, 1) < 0.8:
        second_of_silence = AudioSegment.silent(duration = random.uniform(0,5))
    else:
        second_of_silence = AudioSegment.silent(duration = random.uniform(0,30))
    return second_of_silence

def synthesize_text(texts: List[str], audio_dir: str):
    '''
    text is an array of texts that can be greater in length than 5000 
    between each element in the array we will add an additional 3 seconds of silence 
    and return the combined audio
    '''
    prev = None
    os.makedirs(f'ignore') # temporary folder to store previous text snippets
    for text in texts: # iterate through text array and add append text snippets back to back
        synthesize_text_english(text, "ignore/output1.mp3") 

        audio1 = AudioSegment.from_file("ignore/output1.mp3", format="mp3")

        if prev == None:
            prev = audio1 + gen_silence()
        else:
            prev = prev + audio1 + gen_silence()

    if prev is not None:
        prev.export(f'{audio_dir}/result.mp3', format="mp3")

    shutil.rmtree("ignore") # deleteing the temporary text array folder
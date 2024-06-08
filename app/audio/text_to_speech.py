from pathlib import Path
from pydub import AudioSegment
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


class AudioService:
    def __init__(self):
        self.client = OpenAI(base_url="https://api.openai.com/v1", api_key=api_key)
        self.speech_file_path = Path(__file__).parent / "speech.wav"  # Altera a extensão para .wav

    def text_to_speech(self, text, voice="nova", speed=0.9):
        response_audio = self.client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            speed=speed,
        )

        temp_path = self.speech_file_path.with_suffix('.mp3')
        with open(temp_path, 'wb') as f:
            f.write(response_audio.content)

        sound = AudioSegment.from_mp3(temp_path)
        sound.export(self.speech_file_path, format="wav", parameters=["-ar", str(44100)])

        os.remove(temp_path)

        return self.speech_file_path
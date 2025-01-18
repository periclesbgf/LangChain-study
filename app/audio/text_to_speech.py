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

    def speech_to_text(self, audio_file_path):
        """
        Converte um arquivo de áudio para texto usando a API de reconhecimento de fala.
        """
        try:
            with open(audio_file_path, 'rb') as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="pt"
                )

            transcribed_text = response.text
            print(f"[INFO] Transcribed text: {transcribed_text}")
            return transcribed_text
        except Exception as e:
            print(f"[ERROR] Failed to transcribe audio: {str(e)}")
            return None

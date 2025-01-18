import asyncio
import websockets
import sounddevice as sd
import numpy as np
import base64
import json
import os
from dotenv import load_dotenv
import pygame
import queue  # Fila thread-safe
import logging

# Configuração de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carregar variáveis de ambiente
load_dotenv()

# Configurações da API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

# Configurações de áudio
RATE = 24000  # 24kHz exigido pela API
CHANNELS = 1  # Mono canal
CHUNK_SIZE = 1024  # Tamanho do chunk de áudio

# Criar uma fila de áudio thread-safe
audio_queue = queue.Queue()

# Função para converter Float32Array para PCM16
def float_to_16bit_pcm(float32_array):
    pcm16_array = np.clip(float32_array, -1, 1) * 32767
    return pcm16_array.astype(np.int16).tobytes()

# Função para reproduzir áudio da resposta
def play_audio(audio_data_base64):
    logging.info("Reproduzindo áudio recebido.")
    pygame.mixer.init()
    audio_data = base64.b64decode(audio_data_base64)
    with open("response.wav", "wb") as f:
        f.write(audio_data)

    pygame.mixer.music.load("response.wav")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Função de callback para capturar áudio
def audio_callback(indata, frames, time, status):
    if status:
        logging.warning(f"Status de áudio: {status}")
    
    logging.info("Capturando áudio do microfone.")
    float32_array = indata[:, 0]  # Captura do canal mono
    pcm_data = float_to_16bit_pcm(float32_array)
    base64_audio = base64.b64encode(pcm_data).decode('utf-8')
    
    # Log para mostrar que o áudio foi processado
    logging.debug(f"Tamanho do chunk de áudio: {len(base64_audio)} bytes.")

    audio_queue.put(base64_audio)  # Colocar o áudio na fila

# Função principal para enviar áudio via WebSocket
async def send_audio(ws):
    while True:
        base64_audio = audio_queue.get()  # Pegar o áudio da fila
        logging.info(f"Enviando áudio: {len(base64_audio)} bytes.")
        await ws.send(json.dumps({
            'type': 'input_audio_buffer.append',
            'audio': base64_audio
        }))
        audio_queue.task_done()

# Função principal para conectar ao WebSocket e gerenciar a comunicação
async def realtime_interaction():
    async with websockets.connect(API_URL, extra_headers={
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }) as ws:
        logging.info("Conectado ao servidor de áudio em tempo real.")

        # Iniciar a captura de áudio
        with sd.InputStream(samplerate=RATE, channels=CHANNELS, callback=audio_callback, blocksize=CHUNK_SIZE):
            logging.info("Capturando áudio do microfone...")
            asyncio.create_task(send_audio(ws))

            while True:
                try:
                    # Receber as mensagens do servidor
                    message = await ws.recv()
                    event = json.loads(message)

                    # Se for um áudio de resposta, tocar o áudio
                    if 'audio' in event:
                        logging.info("Áudio recebido do servidor.")
                        play_audio(event['audio'])
                    elif 'text' in event:
                        logging.info(f"Texto recebido: {event['text']}")
                        
                except websockets.ConnectionClosed:
                    logging.error("Conexão com o servidor foi fechada.")
                    break

# Executa a função de interação
asyncio.run(realtime_interaction())

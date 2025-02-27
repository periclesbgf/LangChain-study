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
import struct
import requests
import uuid

# Configuração de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carregar variáveis de ambiente
load_dotenv()

# Configurações da API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SESSION_URL = "https://api.openai.com/v1/realtime/sessions"
WS_API_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"

# Configurações de áudio
RATE = 24000       # 24kHz exigido pela API
CHANNELS = 1       # Canal mono
CHUNK_SIZE = 1024  # Tamanho do chunk de áudio

# Fila para gerenciamento dos chunks de áudio
audio_queue = queue.Queue()

def create_session():
    """
    Cria uma sessão via REST para obter um token efêmero.
    """
    payload = {
        "model": "gpt-4o-realtime-preview-2024-12-17",
        "modalities": ["audio", "text"],
        "instructions": "You are a friendly assistant."
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    logging.info("Criando sessão na API Realtime...")
    response = requests.post(SESSION_URL, headers=headers, json=payload)
    if response.status_code == 200:
        session_data = response.json()
        logging.info("Sessão criada com sucesso:\n%s", json.dumps(session_data, indent=2))
        return session_data
    else:
        logging.error("Erro ao criar sessão: %s", response.text)
        return None

def float_to_16bit_pcm(float32_array):
    """
    Converte um array de float32 para bytes de áudio PCM de 16 bits.
    """
    pcm16_array = np.clip(float32_array, -1, 1) * 32767
    return pcm16_array.astype(np.int16).tobytes()

def add_wav_header(pcm_data, sample_rate, channels):
    """
    Adiciona cabeçalho WAV aos dados PCM, caso eles não possuam.
    """
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_data)
    header = b'RIFF'
    header += struct.pack('<I', 36 + data_size)
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<I', 16)
    header += struct.pack('<H', 1)
    header += struct.pack('<H', channels)
    header += struct.pack('<I', sample_rate)
    header += struct.pack('<I', byte_rate)
    header += struct.pack('<H', block_align)
    header += struct.pack('<H', bits_per_sample)
    header += b'data'
    header += struct.pack('<I', data_size)
    return header + pcm_data

def play_audio(audio_data_base64):
    """
    Reproduz áudio decodificado (base64). Se necessário, adiciona o cabeçalho WAV.
    """
    logging.info("Reproduzindo áudio recebido.")
    audio_data = base64.b64decode(audio_data_base64)
    if audio_data[:4] != b'RIFF':
        logging.info("Áudio recebido sem cabeçalho WAV; adicionando cabeçalho.")
        audio_data = add_wav_header(audio_data, RATE, CHANNELS)
    
    wav_filename = "response.wav"
    with open(wav_filename, "wb") as f:
        f.write(audio_data)
    
    try:
        pygame.mixer.music.load(wav_filename)
        pygame.mixer.music.play()
    except Exception as e:
        logging.error(f"Erro ao reproduzir áudio: {e}")
    
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def audio_callback(indata, frames, time, status):
    """
    Callback chamado pelo sounddevice para cada chunk de áudio capturado.
    Converte o áudio para PCM16 e o codifica em base64 para envio.
    """
    if status:
        logging.warning(f"Status de áudio: {status}")
    logging.info("Capturando áudio do microfone...")
    float32_array = indata[:, 0]  # Usa apenas o canal mono
    pcm_data = float_to_16bit_pcm(float32_array)
    base64_audio = base64.b64encode(pcm_data).decode('utf-8')
    logging.debug(f"Chunk de áudio (base64): {len(base64_audio)} bytes.")
    audio_queue.put(base64_audio)

async def send_audio(ws):
    """
    Envia os chunks de áudio capturados ao servidor via o evento input_audio_buffer.append.
    """
    while True:
        base64_audio = audio_queue.get()
        event = {
            "event_id": str(uuid.uuid4()),
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }
        logging.info(f"Enviando chunk de áudio ({len(base64_audio)} bytes).")
        await ws.send(json.dumps(event))
        audio_queue.task_done()

async def commit_audio_buffer(ws):
    """
    Aguarda o usuário pressionar Enter para enviar o evento input_audio_buffer.commit,
    que comita o áudio enviado (criando um novo item de mensagem do usuário).
    """
    while True:
        # Usa asyncio.to_thread para não bloquear o event loop com input()
        await asyncio.to_thread(input, "Pressione Enter para COMMITAR o áudio (input_audio_buffer.commit)...")
        event = {
            "event_id": str(uuid.uuid4()),
            "type": "input_audio_buffer.commit"
        }
        await ws.send(json.dumps(event))
        logging.info("Evento input_audio_buffer.commit enviado.")

async def realtime_interaction(ephemeral_token):
    """
    Estabelece a conexão WebSocket com o servidor Realtime usando o token efêmero.
    Envia um evento inicial (simulando o exemplo response.create) e inicia:
      - Captura de áudio (e envio dos chunks)
      - Rotina para commit manual do áudio
      - Processamento de eventos recebidos do servidor
    """
    async with websockets.connect(WS_API_URL, extra_headers={
        "Authorization": f"Bearer {ephemeral_token}",
        "OpenAI-Beta": "realtime=v1",
    }) as ws:
        logging.info("Conectado ao servidor Realtime.")
        
        # Envia um evento inicial semelhante ao response.create (pode ser adaptado conforme necessário)
        init_event = {
            "event_id": str(uuid.uuid4()),
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "instructions": "Please assist the user.",
                "voice": "sage",
                "output_audio_format": "pcm16"
            }
        }
        await ws.send(json.dumps(init_event))
        logging.info("Evento response.create enviado.")
        
        # Inicia a captura de áudio do microfone
        with sd.InputStream(samplerate=RATE, channels=CHANNELS, callback=audio_callback, blocksize=CHUNK_SIZE):
            logging.info("Capturando áudio do microfone...")
            # Cria tarefas assíncronas para enviar áudio e para commit manual do buffer
            send_task = asyncio.create_task(send_audio(ws))
            commit_task = asyncio.create_task(commit_audio_buffer(ws))
            
            while True:
                try:
                    message = await ws.recv()
                    data = json.loads(message)
                    logging.info("Evento recebido:\n%s", json.dumps(data, indent=2))
                    
                    if data.get("type") == "input_audio_buffer.committed":
                        logging.info("O áudio foi COMMITADO; item criado: %s", data.get("item_id"))
                    elif data.get("type") == "conversation.item.created":
                        logging.info("Novo item de conversa criado: %s", data.get("item", {}).get("id"))
                    elif data.get("type") == "response.created":
                        logging.info("Resposta em progresso criada: %s", data.get("response", {}).get("id"))
                    elif data.get("type") == "response.done":
                        logging.info("Resposta concluída: %s", data.get("response", {}).get("id"))
                    elif data.get("type") == "error":
                        logging.error("Erro recebido: %s", data.get("error"))
                    # Você pode tratar outros tipos de evento conforme necessário...
                    
                except websockets.ConnectionClosed:
                    logging.error("Conexão com o servidor encerrada.")
                    break

def test_microfone():
    """
    Grava 3 segundos de áudio e o reproduz para testar se o microfone está funcionando.
    """
    logging.info("Testando microfone... Grave 3 segundos de áudio.")
    duration = 3  # segundos
    recording = sd.rec(int(duration * RATE), samplerate=RATE, channels=CHANNELS)
    sd.wait()
    logging.info("Reproduzindo áudio gravado; verifique se o som está correto.")
    sd.play(recording, RATE)
    sd.wait()
    logging.info("Teste do microfone concluído.")

if __name__ == '__main__':
    # Inicializa o pygame.mixer apenas uma vez, com os parâmetros compatíveis
    try:
        pygame.mixer.init(frequency=RATE, channels=CHANNELS)
    except Exception as e:
        logging.error(f"Erro ao inicializar o pygame.mixer: {e}")
    
    # Testa o microfone
    test_microfone()
    input("Microfone OK? Pressione Enter para continuar...")

    # Cria a sessão para obter o token efêmero
    session_data = create_session()
    if session_data is None:
        logging.error("Falha ao criar sessão. Encerrando.")
        exit(1)
    
    ephemeral_token = session_data.get("client_secret", {}).get("value")
    if not ephemeral_token:
        logging.error("Token efêmero não encontrado na resposta da sessão. Encerrando.")
        exit(1)
    
    logging.info("Usando token efêmero para autenticação: %s", ephemeral_token)
    
    # Inicia a interação em tempo real usando o token efêmero
    asyncio.run(realtime_interaction(ephemeral_token))

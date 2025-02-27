import asyncio
import websockets
import pyaudio
import base64
import json
import threading
import queue

class AudioStreamer:
    def __init__(self, websocket_url="ws://127.0.0.1:8000/ws/test-voice"):
        self.websocket_url = websocket_url
        self.is_running = False
        self.audio_queue = queue.Queue()
        
        # Configurações do PyAudio
        self.chunk = 4096
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 44100
        
        self.p = None
        self.stream = None

    def initialize_audio(self):
        """Inicializa o PyAudio e configura o stream"""
        if self.p is None:
            self.p = pyaudio.PyAudio()
            
        if self.stream is None:
            def callback(in_data, frame_count, time_info, status):
                if self.is_running:
                    self.audio_queue.put(in_data)
                return (in_data, pyaudio.paContinue)
            
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=callback
            )

    def start_streaming(self):
        """Inicia o streaming de áudio"""
        self.is_running = True
        self.initialize_audio()
        self.stream.start_stream()
        print("* Streaming iniciado")

    def stop_streaming(self):
        """Para o streaming"""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.p:
            self.p.terminate()
            self.p = None
        print("* Streaming finalizado")

async def audio_sender(streamer, websocket):
    """Função assíncrona para enviar áudio pelo websocket"""
    while streamer.is_running:
        try:
            # Pega o áudio da fila com timeout para não bloquear
            try:
                audio_data = streamer.audio_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.001)  # Pequena pausa para não sobrecarregar
                continue

            # Converte para base64
            base64_audio = base64.b64encode(audio_data).decode('utf-8')
            
            # Cria e envia a mensagem
            message = {
                "type": "audio",
                "data": base64_audio
            }
            await websocket.send(json.dumps(message))
        except Exception as e:
            print(f"Erro ao enviar áudio: {e}")
            break

async def main():
    streamer = AudioStreamer()
    
    try:
        async with websockets.connect(streamer.websocket_url) as websocket:
            print("Conectado ao WebSocket!")
            print("Iniciando streaming de áudio...")
            
            streamer.start_streaming()
            
            # Inicia o sender em uma task separada
            sender_task = asyncio.create_task(audio_sender(streamer, websocket))
            
            try:
                # Mantém a conexão ativa e monitora mensagens do servidor
                while streamer.is_running:
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        print(f"Mensagem recebida: {msg}")
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        print("Conexão fechada")
                        break
            except Exception as e:
                print(f"Erro na conexão: {e}")
            finally:
                # Garante que a task do sender seja cancelada
                sender_task.cancel()
                try:
                    await sender_task
                except asyncio.CancelledError:
                    pass
    finally:
        streamer.stop_streaming()
        print("Programa finalizado")

if __name__ == "__main__":
    # Configura e executa o event loop corretamente
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPrograma encerrado pelo usuário")
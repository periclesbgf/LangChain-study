import asyncio
import websockets
import json
import sys
import argparse

async def test_chat_streaming():
    """
    Test the chat streaming WebSocket endpoint.
    """
    parser = argparse.ArgumentParser(description='Test chat streaming WebSocket.')
    parser.add_argument('--session_id', default='test-session-123', help='Session ID')
    parser.add_argument('--user_email', default='test@example.com', help='User email')
    parser.add_argument('--discipline_id', default='CS101', help='Discipline ID')
    parser.add_argument('--message', default='Explique o conceito de redes de computadores', help='Chat message')
    parser.add_argument('--uri', default='ws://localhost:8000/ws/chat/', help='WebSocket URI base')
    
    args = parser.parse_args()
    
    # Connect to the WebSocket
    uri = f"{args.uri}{args.session_id}"
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            # Wait for initial connection message
            response = await websocket.recv()
            print(f"Connection established: {response}")
            
            # Prepare the message
            message = {
                "type": "chat",
                "user_email": args.user_email,
                "discipline_id": args.discipline_id,
                "message": args.message
            }
            
            # Send the message
            print(f"Sending message: {args.message}")
            await websocket.send(json.dumps(message))
            
            # Receive streaming responses
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                
                # Display based on chunk type
                if data["type"] == "chunk":
                    print(data["content"], end="", flush=True)
                elif data["type"] == "processing":
                    print(f"\n[PROCESSING] {data['content']}")
                elif data["type"] == "complete":
                    print(f"\n\n[COMPLETE] {data['content']}")
                    break
                elif data["type"] == "error":
                    print(f"\n[ERROR] {data['content']}")
                    break
    
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")
    except Exception as e:
        print(f"Error: {e}")

async def test_regular_chat():
    """
    Test the regular chat API endpoint (non-streaming).
    """
    import aiohttp
    
    parser = argparse.ArgumentParser(description='Test regular chat API endpoint.')
    parser.add_argument('--session_id', default='test-session-123', help='Session ID')
    parser.add_argument('--user_email', default='test@example.com', help='User email')
    parser.add_argument('--discipline_id', default='CS101', help='Discipline ID')
    parser.add_argument('--message', default='Explique o conceito de redes de computadores', help='Chat message')
    parser.add_argument('--uri', default='http://localhost:8000/chat', help='Chat API URI')
    
    args = parser.parse_args()
    
    print(f"Sending request to {args.uri}...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Prepare form data
            data = {
                'session_id': args.session_id,
                'discipline_id': args.discipline_id,
                'message': args.message
            }
            
            # Send request
            start_time = asyncio.get_event_loop().time()
            
            async with session.post(args.uri, data=data) as response:
                response_json = await response.json()
                end_time = asyncio.get_event_loop().time()
                
                print(f"Response received in {end_time - start_time:.2f} seconds:")
                print(f"Status: {response.status}")
                print(f"Response: {json.dumps(response_json, indent=2)}")
    
    except Exception as e:
        print(f"Error: {e}")

# Save the original audio streaming test as a separate function
async def test_audio_streaming():
    """
    The original audio streaming test functionality
    """
    import pyaudio
    import base64
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
    # Determine which test to run based on command-line argument
    parser = argparse.ArgumentParser(description='Test chat and audio functionality.')
    parser.add_argument('--mode', choices=['stream', 'regular', 'audio'], default='stream',
                        help='Test mode: stream for WebSocket chat, regular for HTTP API chat, or audio for voice streaming')
    
    args, remaining = parser.parse_known_args()
    
    # Reset sys.argv for the next parser
    sys.argv = [sys.argv[0]] + remaining
    
    try:
        if args.mode == 'stream':
            print("Testing streaming chat WebSocket...")
            asyncio.run(test_chat_streaming())
        elif args.mode == 'regular':
            print("Testing regular chat API endpoint...")
            asyncio.run(test_regular_chat())
        else: # args.mode == 'audio'
            print("Testing audio streaming...")
            asyncio.run(test_audio_streaming())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
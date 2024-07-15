import paho.mqtt.client as mqtt
import ssl
import json
import os
from mqtt.constants import (
    MQTT_BROKER, MQTT_PORT, CA_CERTIFICATE, CLIENT_CERTIFICATE, CLIENT_PRIVATE_KEY,
    MQTT_TOPIC_READ_PORTA, MQTT_TOPIC_READ_WATER_INFO,
    MQTT_TOPIC_READ_DOOR, MQTT_TOPIC_READ_RV_INFO, MQTT_TOPIC_READ_RV_PUMP
)
import threading

# Dicionário para armazenar os dados coletados do MQTT
mqtt_data_store = {
    "door": None,
    "open_door": None,
    "water_info": None,
    "rv_info": None,
    "rv_pump": None
}

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(MQTT_TOPIC_READ_WATER_INFO)
    client.subscribe(MQTT_TOPIC_READ_DOOR)
    client.subscribe(MQTT_TOPIC_READ_PORTA)
    client.subscribe(MQTT_TOPIC_READ_RV_INFO)
    client.subscribe(MQTT_TOPIC_READ_RV_PUMP)

def on_message(client, userdata, msg):
    print(f"Received message: {msg.topic} {str(msg.payload)}")
    try:
        data = json.loads(msg.payload.decode())
        if msg.topic == MQTT_TOPIC_READ_WATER_INFO:
            print("Received water info")
            handle_water_info(data)
        elif msg.topic == MQTT_TOPIC_READ_DOOR:
            handle_esp32_door(data)
        elif msg.topic == MQTT_TOPIC_READ_PORTA:
            handle_esp32_open_door(data)
        elif msg.topic == MQTT_TOPIC_READ_RV_INFO:
            handle_rv_info(data)
        elif msg.topic == MQTT_TOPIC_READ_RV_PUMP:
            handle_rv_pump(data)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON message: {e}")

def handle_water_info(data):
    global mqtt_data_store
    if 'agua' in data and 'cisterna' in data and 'caixa' in data:
        mqtt_data_store["water_info"] = data
        print(f"Water Info - Agua: {data['agua']}, Cisterna: {data['cisterna']}, Caixa: {data['caixa']}")
    else:
        print("Invalid data format for water/info")

def handle_esp32_door(data):
    global mqtt_data_store
    if 'time' in data and 'sensor_porta' in data:
        mqtt_data_store["door"] = data
        print(f"ESP32 Door - Time: {data['time']}, Sensor Porta: {data['sensor_porta']}")
    else:
        print("Invalid data format for esp32/door")

def handle_esp32_open_door(data):
    global mqtt_data_store
    if 'name' in data:
        mqtt_data_store["open_door"] = data
        print(f"ESP32 Open Door - Name: {data['name']}")
    else:
        print("Invalid data format for esp32/open_door")

def handle_rv_info(data):
    global mqtt_data_store
    if all(k in data for k in ('water_level', 'temperatura', 'umidade', 'pressao_barometrica', 'presenca_gas')):
        mqtt_data_store["rv_info"] = data
        print(f"RV Info - Water Level: {data['water_level']}, Temperatura: {data['temperatura']}, Umidade: {data['umidade']}, Pressão Barométrica: {data['pressao_barometrica']}, Presença de Gás: {data['presenca_gas']}")
    else:
        print("Invalid data format for rv/info")

def handle_rv_pump(data):
    global mqtt_data_store
    if 'state_pump' in data:
        mqtt_data_store["rv_pump"] = data
        print(f"RV Pump - State: {data['state_pump']}")
    else:
        print("Invalid data format for rv/pump")

def mqtt_publish(message, topic):
    client = mqtt.Client()
    client.on_connect = on_connect

    client.tls_set(ca_certs=CA_CERTIFICATE,
                   certfile=CLIENT_CERTIFICATE,
                   keyfile=CLIENT_PRIVATE_KEY,
                   tls_version=ssl.PROTOCOL_TLSv1_2)

    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    info = client.publish(topic, message)
    info.wait_for_publish()

    print(f"Message published: MID={info.mid}, Granted QoS={info.rc}")

    client.loop_stop()
    client.disconnect()

def initialize_mqtt_client():
    client = mqtt.Client()

    client.tls_set(ca_certs=os.path.abspath(CA_CERTIFICATE),
                   certfile=os.path.abspath(CLIENT_CERTIFICATE),
                   keyfile=os.path.abspath(CLIENT_PRIVATE_KEY),
                   tls_version=ssl.PROTOCOL_TLSv1_2)
    print("Connecting to MQTT broker")
    client.on_connect = on_connect
    client.on_message = on_message

    mqtt_broker_host = MQTT_BROKER.replace("mqtts://", "").split(":")[0]

    client.connect(mqtt_broker_host, MQTT_PORT, 60)

    return client

def start_mqtt_loop():
    print("Starting MQTT loop")
    mqtt_client = initialize_mqtt_client()
    mqtt_client.loop_forever()

mqtt_thread = threading.Thread(target=start_mqtt_loop)
mqtt_thread.start()

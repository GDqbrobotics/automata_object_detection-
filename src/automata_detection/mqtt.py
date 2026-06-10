import json

import paho.mqtt.client as mqtt


def mqtt_send(*, topic: str, host: str, port: int, send_queue, username: str = "", password: str = "", verbose: bool = False) -> None:
    def on_connect(client, userdata, flags, rc):
        print("[MQTT] Connected to broker with result code", rc)

    client = mqtt.Client()
    if username or password:
        client.username_pw_set(username, password)

    client.on_connect = on_connect
    client.connect(host, port, 60)
    client.loop_start()

    while True:
        event = send_queue.get()
        if verbose:
            print("[MQTT] Sending event:", event)
        client.publish(topic, json.dumps(event))

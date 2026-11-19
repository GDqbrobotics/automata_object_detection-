import json
import threading
import time

import paho.mqtt.client as mqtt


def mqtt_send(*, topic: str, host: str, port: int, send_queue, username: str = "", password: str = "", verbose: bool = False) -> None:
    """MQTT publisher process - sends messages from send_queue to the specified topic."""
    def on_connect(client, userdata, flags, rc):
        print(f"[MQTT Publisher] Connected to broker with result code {rc}")

    def on_disconnect(client, userdata, rc):
        print(f"[MQTT Publisher] Disconnected with result code {rc}")

    client = mqtt.Client()
    if username or password:
        client.username_pw_set(username, password)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    
    client.connect(host, port, 60)
    client.loop_start()

    try:
        while True:
            event = send_queue.get()
            if event is None:  # Shutdown signal
                break
            if verbose:
                print("[MQTT Publisher] Sending event:", event)
            client.publish(topic, json.dumps(event), qos=1)
    finally:
        client.loop_stop()
        client.disconnect()


def mqtt_listen(*, host: str, port: int, command_queue, subscribe_topic: str, username: str = "", password: str = "", verbose: bool = False) -> None:
    """MQTT subscriber process - listens for commands on subscribe_topic and puts them in command_queue."""
    def on_connect(client, userdata, flags, rc):
        print(f"[MQTT Subscriber] Connected to broker with result code {rc}")
        client.subscribe(subscribe_topic, qos=1)
        print(f"[MQTT Subscriber] Subscribed to topic: {subscribe_topic}")

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if verbose:
                print(f"[MQTT Subscriber] Received on {msg.topic}: {payload}")
            command_queue.put(payload)
        except json.JSONDecodeError as e:
            print(f"[MQTT Subscriber] Failed to decode JSON: {e}")
        except Exception as e:
            print(f"[MQTT Subscriber] Error processing message: {e}")

    def on_disconnect(client, userdata, rc):
        print(f"[MQTT Subscriber] Disconnected with result code {rc}")

    client = mqtt.Client()
    if username or password:
        client.username_pw_set(username, password)

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    
    client.connect(host, port, 60)
    client.loop_forever()


class MQTTClient:
    """Threaded MQTT client for use within the inference process (single process mode)."""
    
    def __init__(self, *, host: str, port: int, status_topic: str, results_topic: str, 
                 command_topic: str, username: str = "", password: str = "", verbose: bool = False):
        self.host = host
        self.port = port
        self.status_topic = status_topic
        self.results_topic = results_topic
        self.command_topic = command_topic
        self.username = username
        self.password = password
        self.verbose = verbose
        
        self.client = mqtt.Client()
        if username or password:
            self.client.username_pw_set(username, password)
            
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        self._command_callback = None
        self._connected = False
        self._lock = threading.Lock()
        
    def _on_connect(self, client, userdata, flags, rc):
        with self._lock:
            self._connected = True
        print(f"[MQTT Client] Connected with result code {rc}")
        client.subscribe(self.command_topic, qos=1)
        print(f"[MQTT Client] Subscribed to: {self.command_topic}")
        
    def _on_disconnect(self, client, userdata, rc):
        with self._lock:
            self._connected = False
        print(f"[MQTT Client] Disconnected with result code {rc}")
        
    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if self.verbose:
                print(f"[MQTT Client] Received on {msg.topic}: {payload}")
            if self._command_callback:
                self._command_callback(payload)
        except json.JSONDecodeError as e:
            print(f"[MQTT Client] Failed to decode JSON: {e}")
        except Exception as e:
            print(f"[MQTT Client] Error processing message: {e}")
            
    def set_command_callback(self, callback):
        """Set callback function to handle incoming commands."""
        self._command_callback = callback
        
    def connect(self):
        self.client.connect(self.host, self.port, 60)
        self.client.loop_start()
        
    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        
    def publish_status(self, status: str):
        """Publish status message to status topic."""
        message = {"state": status}
        self.client.publish(self.status_topic, json.dumps(message), qos=1)
        if self.verbose:
            print(f"[MQTT Client] Published status: {status}")
            
    def publish_results(self, results: list):
        """Publish results message to results topic - one message per object."""
        for obj in results:
            obj_id = obj.get("object_number", 0)
            # Convert from mm to meters
            x1 = obj.get("x_1", 0) / 1000.0
            y1 = obj.get("y_1", 0) / 1000.0
            z1 = obj.get("z_1", 0) / 1000.0
            x2 = obj.get("x_2", 0) / 1000.0
            y2 = obj.get("y_2", 0) / 1000.0
            z2 = obj.get("z_2", 0) / 1000.0
            
            message = {
                "type": "object_points",
                "object_id": f"object_{obj_id:03d}",
                "points": [[x1, y1, z1], [x2, y2, z2]]
            }
            self.client.publish(self.results_topic, json.dumps(message), qos=1)
            if self.verbose:
                print(f"[MQTT Client] Published result for {message['object_id']}: {message['points']}")
    
    def publish_objects_detected(self, object_ids: list):
        """Publish objects_detected message with list of detected objects."""
        message = {
            "type": "objects_detected",
            "objects": [f"object_{obj_id:03d}" for obj_id in object_ids],
            "count": len(object_ids)
        }
        self.client.publish(self.results_topic, json.dumps(message), qos=1)
        if self.verbose:
            print(f"[MQTT Client] Published objects_detected: {message}")
            
    def is_connected(self) -> bool:
        with self._lock:
            return self._connected
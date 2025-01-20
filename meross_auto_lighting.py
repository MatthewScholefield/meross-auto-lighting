import asyncio
from contextlib import suppress
from collections import namedtuple
from enum import Enum
from datetime import datetime, time
from typing import List, Tuple, Optional, Union
from argparse import ArgumentParser
from loguru import logger
from pathlib import Path
from pydantic import BaseModel
import yaml
import paho.mqtt.client as mqtt
import threading

from meross_iot.device_factory import BaseDevice
from meross_iot.http_api import MerossHttpClient
from meross_iot.manager import MerossManager
from meross_iot.controller.mixins.light import LightMixin
from aiologic.lowlevel import AsyncEvent

class RgbColor(BaseModel):
    rgb: Tuple[int, int, int]

    @property
    def is_off(self) -> bool:
        return not any(self.rgb)


class WhiteColor(BaseModel):
    luminance: int
    temperature: int

    @property
    def is_off(self) -> bool:
        return self.luminance == 0


class LightStateConfig(BaseModel):
    name: str
    start: time
    lights: List[Union[RgbColor, WhiteColor]]


OFF_COLOR = WhiteColor(luminance=0, temperature=100)
AWAY_CONFIG = LightStateConfig(name='Away', start=time(0, 0), lights=[OFF_COLOR, OFF_COLOR])


class AppConfig(BaseModel):
    meross_email: str
    meross_password: str
    user_ips: List[str]
    presence_timeout: int = 1800
    light_states: List[LightStateConfig]
    mqtt_host: str
    mqtt_port: int = 1883
    mqtt_topic: str
    mqtt_publish_topic: str
    mqtt_user: str
    mqtt_password_file: Path

    @staticmethod
    def load_from_file(filepath: str) -> 'AppConfig':
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return AppConfig(**data)

    def get_current_state_index(self) -> int:
        now = datetime.now().time()
        for i, config in enumerate(self.light_states):
            end_time = self.light_states[(i + 1) % len(self.light_states)].start
            if config.start < end_time:
                if config.start <= now < end_time:
                    return i
            else:
                if now >= config.start or now < end_time:
                    return i
        raise RuntimeError('No state in config matches current time!')


class DeviceState(str, Enum):
    ONLINE = 'online'
    RECENT_OFFLINE = 'recent_offline'
    OFFLINE = 'offline'


class DeviceTracker:
    def __init__(self, ips: List[str], seen_timeout_seconds: int):
        self.ips = ips
        self.seen_timeout_seconds = seen_timeout_seconds
        self.last_seen: Optional[datetime] = None
        self.last_state: Optional[DeviceState] = None
        self.is_present = True

    async def is_device_present(self) -> bool:
        state, is_new = await self.check_device_state()
        if is_new:
            if state == DeviceState.ONLINE:
                self.is_present = True
            elif state == DeviceState.OFFLINE:
                self.is_present = False
        return self.is_present

    async def check_device_state(self) -> Tuple[DeviceState, bool]:
        current_state = await self._get_current_state()
        is_new = current_state != self.last_state
        self.last_state = current_state
        if current_state == DeviceState.ONLINE:
            self.last_seen = datetime.now()
        return current_state, is_new

    async def _get_current_state(self) -> DeviceState:
        device_statuses = await asyncio.gather(*[self._is_device_up(ip) for ip in self.ips])
        if any(device_statuses):
            return DeviceState.ONLINE
        if (
            self.last_seen
            and (datetime.now() - self.last_seen).total_seconds()
            < self.seen_timeout_seconds
        ):
            return DeviceState.RECENT_OFFLINE
        return DeviceState.OFFLINE

    async def _is_device_up(self, ip: str) -> bool:
        process = await asyncio.create_subprocess_exec(
            'ping', '-c', '1', '-W', '0.5', ip,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return process.returncode == 0


class LightDeviceType(BaseDevice, LightMixin):
    pass


class LightManager:
    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
        self.http_api_client: Optional[MerossHttpClient] = None
        self.manager: Optional[MerossManager] = None
        self.devices: List[LightDeviceType] = []
        self.current_colors: List[Optional[Union[RgbColor, WhiteColor]]] = []

    async def __aenter__(self):
        self.http_api_client = await MerossHttpClient.async_from_user_password(
            email=self.email, password=self.password, api_base_url='https://iot.meross.com',
        )
        self.manager = MerossManager(http_client=self.http_api_client)
        await self.manager.async_init()
        await self.manager.async_device_discovery()
        self.devices = self.manager.find_devices(device_class=LightMixin)
        self.devices.sort(key=lambda dev: dev.name)
        logger.info('Found devices: {}', [dev.name for dev in self.devices])
        for dev in self.devices:
            await dev.async_update()
        self.current_colors = [None] * len(self.devices)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.manager:
            self.manager.close()
        if self.http_api_client:
            await self.http_api_client.async_logout()

    async def set_light_colors(self, colors: List[Union[RgbColor, WhiteColor]]):
        if self.current_colors == colors:
            return
        await asyncio.gather(
            *[self._set_device_light_color(i, color) for i, color in enumerate(colors)]
        )

    async def _set_device_light_color(self, index: int, color: Union[RgbColor, WhiteColor]):
        if color.is_off:
            await self.devices[index].async_turn_off()
            self.current_colors[index] = None
            return
        args = {}
        if isinstance(color, RgbColor):
            args['rgb'] = color.rgb
            await self.devices[index].async_set_light_color(luminance=100, temperature=50)
        elif isinstance(color, WhiteColor):
            args['luminance'] = color.luminance
            args['temperature'] = color.temperature
        await self.devices[index].async_set_light_color(**args)
        self.current_colors[index] = color


LightOverride = namedtuple('LightOverride', 'base_index offset')


class MqttLightController:
    def __init__(self, config: AppConfig):
        self.config = config
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.override: Optional[LightOverride] = None
        self.change_event: Optional[AsyncEvent] = None

    def on_connect(self, client, userdata, flags, rc):
        logger.info('Connected to MQTT broker.')
        client.subscribe(self.config.mqtt_topic)

    def on_message(self, client, userdata, msg):
        try:
            value = int(msg.payload.decode())
            self.override = LightOverride(self.config.get_current_state_index(), value) if value else None
            self.change_event.set()
        except ValueError:
            logger.warning("Invalid MQTT message: {}", msg.payload)

    def publish_current_state(self, config_name: str):
        self.mqtt_client.publish(self.config.mqtt_publish_topic, config_name)

    def start(self):
        self.mqtt_client.username_pw_set(self.config.mqtt_user, self.config.mqtt_password_file.read_text().strip())
        self.mqtt_client.connect(self.config.mqtt_host, self.config.mqtt_port)
        threading.Thread(target=self.mqtt_client.loop_forever, daemon=True).start()

    async def run(self):
        self.change_event = AsyncEvent()
        self.start()
        device = DeviceTracker(self.config.user_ips, self.config.presence_timeout)
        async with LightManager(self.config.meross_email, self.config.meross_password) as light_manager:
            last_light_config = None
            while True:
                index = self.config.get_current_state_index()
                if self.override:
                    if index != self.override.base_index:
                        self.override = None
                        self.mqtt_client.publish(self.config.mqtt_topic, str(0), retain=True)
                    else:
                        index = (index + self.override.offset) % len(self.config.light_states)
                is_present = await device.is_device_present()
                light_config = self.config.light_states[index] if is_present else AWAY_CONFIG
                if last_light_config != light_config:
                    last_light_config = light_config
                    logger.info('Setting lights to config: {}', light_config.name)
                    await light_manager.set_light_colors(light_config.lights)
                    self.publish_current_state(light_config.name)
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.change_event, timeout=10)


def main():
    parser = ArgumentParser(description='Smart Lighting Controller with MQTT Support')
    parser.add_argument('config', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()

    config = AppConfig.load_from_file(args.config)
    controller = MqttLightController(config)

    try:
        asyncio.run(controller.run())
    except KeyboardInterrupt:
        print('Shutting down...')


if __name__ == '__main__':
    main()

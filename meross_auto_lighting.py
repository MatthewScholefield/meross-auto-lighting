import asyncio
import subprocess
from enum import Enum
from pathlib import Path
from datetime import datetime, time
from typing import List, Tuple, Optional, Union
import yaml
from meross_iot.device_factory import BaseDevice
from meross_iot.http_api import MerossHttpClient
from meross_iot.manager import MerossManager
from meross_iot.controller.mixins.light import LightMixin
from argparse import ArgumentParser
from pydantic import BaseModel
from loguru import logger


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
    user_ip: str
    presence_timeout: int = 1800
    light_states: List[LightStateConfig]
    mqtt_host: str
    mqtt_port: int
    mqtt_user: str
    mqtt_password_file: str
    mqtt_override_topic: str
    mqtt_state_topic: str

    @staticmethod
    def load_from_file(filepath: str) -> 'AppConfig':
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return AppConfig(**data)

    def get_current_light_state(self) -> LightStateConfig:
        now = datetime.now().time()
        for i, config in enumerate(self.light_states):
            end_time = self.light_states[(i + 1) % len(self.light_states)].start
            if config.start < end_time:
                if config.start <= now < end_time:
                    return config
            else:
                if now >= config.start or now < end_time:
                    return config
        raise RuntimeError('No state in config matches current time!')


class DeviceState(str, Enum):
    ONLINE = 'online'
    RECENT_OFFLINE = 'recent_offline'
    OFFLINE = 'offline'


class DeviceTracker:
    def __init__(self, ip: str, seen_timeout_seconds: int):
        self.ip = ip
        self.seen_timeout_seconds = seen_timeout_seconds
        self.last_seen: Optional[datetime] = None
        self.last_state: Optional[DeviceState] = None

    async def check_device_state(self) -> Tuple[DeviceState, bool]:
        current_state = await self._get_current_state()
        is_new = current_state != self.last_state
        self.last_state = current_state
        if current_state == DeviceState.ONLINE:
            self.last_seen = datetime.now()
        return current_state, is_new

    async def _get_current_state(self) -> DeviceState:
        process = await asyncio.create_subprocess_exec(
            'ping', '-c', '1', '-W', '0.5', self.ip,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            return DeviceState.ONLINE
        if self.last_seen and (datetime.now() - self.last_seen).total_seconds() < self.seen_timeout_seconds:
            return DeviceState.RECENT_OFFLINE
        return DeviceState.OFFLINE


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
            email=self.email, password=self.password, api_base_url='https://iot.meross.com'
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
            await self.devices[index].async_set_light_color(rgb=(255, 255, 255))
            args['luminance'] = color.luminance
            args['temperature'] = color.temperature
        await self.devices[index].async_set_light_color(**args)
        self.current_colors[index] = color

    async def turn_off(self):
        for i, device in enumerate(self.devices):
            if self.current_colors[i] and not self.current_colors[i].is_off:
                await device.async_turn_off()
                self.current_colors[i] = None


class LightController:
    def __init__(self, config: AppConfig):
        self.config = config
        self.tracker = DeviceTracker(self.config.user_ip, self.config.presence_timeout)
        self.current_override: Optional[int] = None
        self.override_base: Optional[LightStateConfig] = None

    async def run(self):
        async with LightManager(self.config.meross_email, self.config.meross_password) as light_manager:
            is_away = False
            last_light_config = None
            while True:
                current_state, is_new = await self.tracker.check_device_state()
                if is_new:
                    is_away = current_state != DeviceState.ONLINE
                light_config = AWAY_CONFIG if is_away else self.config.get_current_light_state()
                if self.current_override is not None and light_config == self.override_base:
                    light_config = self._apply_override(light_config)
                elif light_config != self.override_base:
                    self.current_override = None
                    self.override_base = None
                if last_light_config != light_config:
                    last_light_config = light_config
                    await light_manager.set_light_colors(light_config.lights)
                    self._publish_state(self.config.mqtt_state_topic, self.config.light_states.index(light_config))
                await asyncio.sleep(10)

    def _apply_override(self, base_config: LightStateConfig) -> LightStateConfig:
        base_index = self.config.light_states.index(base_config)
        override_index = (base_index + self.current_override) % len(self.config.light_states)
        return self.config.light_states[override_index]

    def _subscribe_override(self):
        result = subprocess.run(
            ['mosquitto_sub', '-h', self.config.mqtt_host, '-p', str(self.config.mqtt_port), '-u', self.config.mqtt_user,
             '-P', Path(self.config.mqtt_password_file).read_text().strip(), '-t', self.config.mqtt_override_topic, '-C', '1'],
            capture_output=True, text=True)
        message = result.stdout.strip()
        try:
            override_value = int(message)
            self.current_override = override_value
            self.override_base = self.config.get_current_light_state()
        except ValueError:
            logger.warning('Invalid override value received: {}', message)

    def _publish_state(self, topic: str, state_index: int):
        subprocess.run(
            ['mosquitto_pub', '-h', self.config.mqtt_host, '-p', str(self.config.mqtt_port), '-u', self.config.mqtt_user,
             '-P', Path(self.config.mqtt_password_file).read_text().strip(), '-t', topic, '-m', str(state_index)],
            check=True)


def main():
    parser = ArgumentParser(description='Smart Lighting Controller')
    parser.add_argument('config', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()
    config = AppConfig.load_from_file(args.config)
    controller = LightController(config)
    try:
        asyncio.run(controller.run())
    except KeyboardInterrupt:
        print('Shutting down...')


if __name__ == '__main__':
    main()

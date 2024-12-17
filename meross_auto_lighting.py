import asyncio
from enum import Enum
from meross_iot.device_factory import BaseDevice
import yaml

from datetime import datetime, time
from typing import List, Tuple, Optional, Union
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
    end: time
    lights: List[Union[RgbColor, WhiteColor]]


class AppConfig(BaseModel):
    meross_email: str
    meross_password: str
    user_ip: str
    presence_timeout: int = 1800
    light_states: List[LightStateConfig]

    @staticmethod
    def load_from_file(filepath: str) -> 'AppConfig':
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return AppConfig(**data)

    def get_current_light_state(self) -> LightStateConfig:
        now = datetime.now().time()
        for config in self.light_states:
            if config.start <= config.end:
                # Normal case: start is before end
                if config.start <= now < config.end:
                    return config
            else:
                # Crosses midnight case
                if now >= config.start or now < config.end:
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
        """Returns (current_state, is_new)"""
        current_state = await self._get_current_state()
        is_new = current_state != self.last_state
        self.last_state = current_state
        if current_state == DeviceState.ONLINE:
            self.last_seen = datetime.now()
        return current_state, is_new

    async def _get_current_state(self) -> DeviceState:
        process = await asyncio.create_subprocess_exec(
            'ping',
            '-c',
            '1',
            '-W',
            '0.5',
            self.ip,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            return DeviceState.ONLINE
        if (
            self.last_seen
            and (datetime.now() - self.last_seen).total_seconds()
            < self.seen_timeout_seconds
        ):
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
        self.is_old_hardware: List[bool] = []
        self.current_colors: List[Optional[Union[RgbColor, WhiteColor]]] = []

    async def __aenter__(self):
        self.http_api_client = await MerossHttpClient.async_from_user_password(
            email=self.email,
            password=self.password,
            api_base_url='https://iot.meross.com',
        )
        self.manager = MerossManager(http_client=self.http_api_client)
        await self.manager.async_init()
        await self.manager.async_device_discovery()
        self.devices: List[LightDeviceType] = self.manager.find_devices(
            device_class=LightMixin
        )
        self.devices.sort(key=lambda dev: dev.name)
        logger.info('Found the following devices:', [dev.name for dev in self.devices])
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

    async def _set_device_light_color(
        self, index: int, color: Union[RgbColor, WhiteColor]
    ):
        if color.is_off:
            await self.devices[index].async_turn_off()
            self.current_colors[index] = None
            return
        logger.info(
            'Setting light color for device {} to: {}', self.devices[index].name, color
        )
        args = {}
        if isinstance(color, RgbColor):
            args['rgb'] = color.rgb
        elif isinstance(color, WhiteColor):
            args['luminance'] = color.luminance
            args['temperature'] = color.temperature
        else:
            assert False
        # Due to strange bug, we need to set a luminance color before an RGB one
        await self.devices[index].async_set_light_color(luminance=100, temperature=50)
        await self.devices[index].async_set_light_color(**args)
        self.current_colors[index] = color

    async def turn_off(self):
        logger.info('Turning off lights')
        for device in self.devices:
            await device.async_turn_off()


class LightController:
    def __init__(self, config: AppConfig):
        self.config = config
        self.tracker = DeviceTracker(self.config.user_ip, self.config.presence_timeout)

    async def run(self):
        async with LightManager(
            self.config.meross_email, self.config.meross_password
        ) as light_manager:
            while True:
                current_state, is_new = await self.tracker.check_device_state()
                if is_new:
                    if current_state == DeviceState.ONLINE:
                        current_state = self.config.get_current_light_state()
                        await light_manager.set_light_colors(current_state.lights)
                    elif current_state == DeviceState.OFFLINE:
                        await light_manager.turn_off()
                await asyncio.sleep(10)


def main():
    parser = ArgumentParser(description='Smart Lighting Controller')
    parser.add_argument('config', type=str, help='Path to the configuration YAML file')
    parser.add_argument(
        'command',
        choices=['run', 'check-device', 'check-light-state'],
        help='Command to execute',
    )

    args = parser.parse_args()
    config = AppConfig.load_from_file(args.config)

    if args.command == 'check-device':
        tracker = DeviceTracker(config.user_ip, config.presence_timeout)
        state, _ = asyncio.run(tracker.check_device_state())
        print('Detected device state:', state)

    elif args.command == 'check-light-state':
        current_state = config.get_current_light_state()
        print(f'Current target light state: {current_state}')

    elif args.command == 'run':
        controller = LightController(config)
        try:
            asyncio.run(controller.run())
        except KeyboardInterrupt:
            print('Shutting down...')


if __name__ == '__main__':
    main()

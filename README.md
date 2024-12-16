# Meross Auto Lighting

This script automatically controls Meross lights based on a lighting schedules and IP based presence detection.

## Features

- Dynamic light adjustment based on network presence
- Scheduled lighting states
- Supports Meross IoT light devices

## Setup and Installation

Meross Auto Lighting uses Rye for dependency management and the development workflow. To get started with development, ensure you have [Rye](https://github.com/astral-sh/rye) installed and then clone the repository and set up the environment:

```sh
git clone https://github.com/MatthewScholefield/meross-auto-lighting.git
cd meross-auto-lighting
rye sync
rye run pre-commit install
```

## Configuration

Create a YAML configuration file with your settings. Below is a sample configuration file (see `config.example.yaml`):

```yaml
meross_email: "your_email@example.com"
meross_password: "your_password"
user_ip: "192.168.1.100"
presence_timeout: 1800
light_states:
  - name: "Morning"
    start: "06:00:00"
    end: "08:00:00"
    lights:
      - rgb: [255, 200, 100]
  - name: "Evening"
    start: "18:00:00"
    end: "22:00:00"
    lights:
      - rgb: [255, 100, 50]
```

## Usage

Run the script with the desired command:

- **Run the main control loop**:
  ```sh
  rye run meross-auto-lighting config.yaml run
  ```

- **Check device state**:
  ```sh
  prye run meross-auto-lighting config.yaml check-device
  ```

- **Check current light state**:
  ```sh
  rye run meross-auto-lighting config.yaml check-light-state
  ```

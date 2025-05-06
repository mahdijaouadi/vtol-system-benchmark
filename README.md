# VTOL System Benchmark

This project implements and compares two different control approaches for a VTOL (Vertical Take-Off and Landing) system simulation: a traditional PID controller and a Reinforcement Learning-based controller. The system simulates a seesaw mechanism with two motors/propellers that can be controlled to maintain a desired angle.

## Project Structure

```
.
├── pid_controller/
│   └── simulation.py      # PID controller implementation and simulation
├── reinforcement learning/
│   ├── simulation.py      # RL-based controller implementation and simulation
│   ├── training/         # Training data and models
│   └── agents/          # RL agent implementations
├── helper_functions/
│   └── tools.py         # Utility functions for data handling
└── labview/            # LabVIEW interface files
```

## Components

### 1. PID Controller Implementation
- Implements a traditional PID controller for angle control
- Features real-time parameter tuning (Kp, Ki, Kd)
- Includes visualization of the seesaw system
- Supports target angle updates via socket communication

### 2. Reinforcement Learning Implementation
- Uses a neural network-based agent for control
- Implements a Deep Q-Learning approach
- Features two agents controlling left and right motors

### 3. System Simulation
Both implementations include:
- Virtual motor and propeller simulation
- Seesaw physics model
- Gyroscope sensor simulation
- Real-time visualization
- Data logging capabilities

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Socket support for communication

## Usage

### PID Controller
1. Navigate to the pid_controller directory
2. Run the simulation:
```bash
python simulation.py
```
3. run LabVIEW dashboard and specify your python execuable path

### Reinforcement Learning Controller
1. Navigate to the reinforcement learning directory
2. Run the simulation:
```bash
python simulation.py
```
3. run LabVIEW dashboard and specify your python execuable path

## Communication Interface

The system uses socket communication for:
- Target angle updates (Port: 65432)
- PID parameter tuning (Ports: 65433-65435)

## Data Logging

The system automatically logs:
- Measured angles
- Target angles

Data is saved in JSON format for easy analysis and visualization.

## Visualization

Both implementations include real-time visualization of:
- Seesaw angle
- Target angle
- System response
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
import socket
import json
import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class Agent(nn.Module):
    def __init__(self,input_size):
        super(Agent, self).__init__()
        self.bn_input=nn.BatchNorm1d(input_size)

        self.fc1=nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2=nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.output_fc=nn.Linear(64, 3)
    def forward(self, x): 
        out=self.bn_input(x)
        out=F.relu(self.fc1(out))
        out=self.bn1(out)
        out=F.relu(self.fc2(out))
        out=self.bn2(out)
        out=self.output_fc(out)
        return out
    def act(self,obs):
        self.eval()
        q_values=self(obs)
        max_q_index=torch.argmax(q_values)
        action=max_q_index.detach().item()
        self.train()
        return action
    

agent1= Agent(input_size=4)
agent2 = Agent(input_size=4)
agent1.load_state_dict(torch.load('agent1_v1.pth'))
agent2.load_state_dict(torch.load('agent2_v1.pth'))

# Virtual Motor+Propeller Class
class VirtualMotorPropeller:
    def __init__(self, R, Ke, Kt, thrust_coefficient, voltage_limit):
        self.angular_velocity = 0.0  # rad/sec
        self.moment_inertia=0.012
        self.R = R
        self.Ke = Ke
        self.Kt = Kt
        self.thrust_coefficient = thrust_coefficient
        self.voltage_limit = voltage_limit
        self.current = 0.0

    def apply_voltage(self, voltage, dt):
        voltage = max(min(voltage, self.voltage_limit), -self.voltage_limit)
        back_emf = self.Ke * self.angular_velocity
        self.current = (voltage - back_emf) / self.R
        torque = self.Kt * self.current
        angular_acceleration = torque/self.moment_inertia
        self.angular_velocity += angular_acceleration * dt


    def get_thrust(self):
        return self.thrust_coefficient * (self.angular_velocity ** 2)
# Seesaw Model Class
class Seesaw:
    def __init__(self, J, arm_length, damping):
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.J = J  # moment of inertia
        self.arm_length = arm_length
        self.damping = damping 

    def apply_forces(self, thrust_left, thrust_right, dt):
        net_torque = (self.arm_length/2) * (thrust_right - thrust_left)
        gravity_torque = 0.3 * 9.8 * (self.arm_length / 2) * np.sin(self.angle)
        total_torque = net_torque - gravity_torque
        angular_acceleration = (total_torque - self.damping * self.angular_velocity) / self.J
        self.angular_velocity += angular_acceleration * dt
        self.angle += self.angular_velocity * dt

    def get_angle_deg(self):
        return self.angle * (180.0 / np.pi)

class Gyroscope:
    def __init__(self, noise_std=0.0):
        self.noise_std = noise_std

    def read_angle(self, true_angle):
        noise = np.random.normal(0, self.noise_std)
        return true_angle + noise

angle_deg_shared = [0]  # Use a list to make it mutable in threads

def update_target_angle():
    global target_angle_deg
    while True:

        HOST = '127.0.0.1'  # Localhost
        PORT = 65432        # Port to listen on
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            conn, addr = s.accept()
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    data = data.decode()
                    new_angle=float(data)
                    
        target_angle_deg = new_angle
        print("Target angle changed to: ",target_angle_deg)
def update_kp():
    while True:

        HOST = '127.0.0.1'  # Localhost
        PORT = 65433       # Port to listen on
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            conn, addr = s.accept()
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    data = data.decode()
                    kp=float(data)
        # pid.kp = kp
        # print("kp coefficient changed to: ",kp)

def update_ki():
    while True:

        HOST = '127.0.0.1'  # Localhost
        PORT = 65434       # Port to listen on
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            conn, addr = s.accept()
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    data = data.decode()
                    ki=float(data)
        # pid.ki = ki
        # print("ki coefficient changed to: ",ki)

def update_kd():
    while True:

        HOST = '127.0.0.1'  # Localhost
        PORT = 65435       # Port to listen on
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            conn, addr = s.accept()
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    data = data.decode()
                    kd=float(data)
        # pid.kd = kd
        # print("kd coefficient changed to: ",kd)

def save_data():
    while True:
        data={
            "measured_angle":gyro.read_angle(seesaw.get_angle_deg()),
            "target_angle": target_angle_deg
        }
        with open("data.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
        time.sleep(2)


def take_action(state):
    state=torch.tensor([state], dtype=torch.float32)

    action1=agent1.act(state)
    action2=agent2.act(state)
    print(action1, action2)
    if action1==0:
        control_signal_right=-12
    elif action1==1:
        control_signal_right=12
    else:
        control_signal_right=0
    
    if action2==0:
        control_signal_left=-12
    elif action2==1:
        control_signal_left=12
    else:
        control_signal_left=0
    return control_signal_right,control_signal_left
# Simulation function (runs in background)
def run_simulation():
    global step
    voltage_left=0
    voltage_right=0
    while True:
        measured_angle_deg = gyro.read_angle(seesaw.get_angle_deg())
        angle_deg_shared[0] = measured_angle_deg  # update shared angle
        state=[target_angle_deg,measured_angle_deg,left_motor.get_thrust(),right_motor.get_thrust()]
        control_signal_right,control_signal_left = take_action(state)
        voltage_right = np.clip(voltage_right+control_signal_right, 0, voltage_limit)
        voltage_left = np.clip(voltage_left+control_signal_left, 0, voltage_limit)

        left_motor.apply_voltage(voltage_left, dt)
        right_motor.apply_voltage(voltage_right, dt)

        thrust_left = left_motor.get_thrust()
        thrust_right = right_motor.get_thrust()

        seesaw.apply_forces(thrust_left, thrust_right, dt)

        # Log data (optional)
        time_log.append(step * dt)
        angle_log.append(measured_angle_deg)
        voltage_left_log.append(voltage_left)
        voltage_right_log.append(voltage_right)
        step += 1

        time.sleep(dt)

# Animation function for matplotlib
def draw_seesaw(i):
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')  # preserve the aspect ratio

    ax.set_title(f"Seesaw Angle: {angle_deg_shared[0]:.2f}° | Target: {target_angle_deg:.2f}°")

    angle_rad = np.radians(angle_deg_shared[0])
    length = 0.8  # total seesaw length

    # Rotate seesaw bar endpoints
    x0, y0 = -length / 2, 0
    x1, y1 = length / 2, 0

    # Apply rotation
    x0_rot = x0 * np.cos(angle_rad) - y0 * np.sin(angle_rad)
    y0_rot = x0 * np.sin(angle_rad) + y0 * np.cos(angle_rad)
    x1_rot = x1 * np.cos(angle_rad) - y1 * np.sin(angle_rad)
    y1_rot = x1 * np.sin(angle_rad) + y1 * np.cos(angle_rad)

    ax.plot([x0_rot, x1_rot], [y0_rot, y1_rot], 'b', linewidth=5)
    ax.plot(0, 0, 'ko')  # pivot

# ---------- Initialization ---------- #
# Parameters
dt = 0.1
target_angle_deg = 23.5

R = 0.5
Ke = 0.02
Kt = 0.02
thrust_coefficient = 1e-5
voltage_limit = 24.0

J = 0.012
arm_length = 0.4
damping = 0.05

left_motor = VirtualMotorPropeller(R, Ke, Kt, thrust_coefficient, voltage_limit)
right_motor = VirtualMotorPropeller(R, Ke, Kt, thrust_coefficient, voltage_limit)
seesaw = Seesaw(J, arm_length, damping)
gyro = Gyroscope(noise_std=0.0)
kp=4
ki=1
kd=1e-1

time_log, angle_log, voltage_left_log, voltage_right_log = [], [], [], []
step = 0

# ---------- Start Threads ---------- #
threading.Thread(target=update_target_angle, daemon=True).start()
threading.Thread(target=update_kp, daemon=True).start()
threading.Thread(target=update_ki, daemon=True).start()
threading.Thread(target=update_kd, daemon=True).start()
threading.Thread(target=save_data, daemon=True).start()

threading.Thread(target=run_simulation, daemon=True).start()

# ---------- Setup Plot ---------- #
fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, draw_seesaw, interval=100)
plt.show()

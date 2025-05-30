import numpy as np
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.e_integral = 0
        self.prev_error = 0

    def update(self, setpoint, measurement, dt):
        error = setpoint - measurement
        self.e_integral += error * dt
        e_derivative = (error - self.prev_error) / dt
        self.prev_error = error

        control = self.Kp * error + self.Ki * self.e_integral + self.Kd * e_derivative
        return control

def aperiodic_system_response(control, dt):
    # Пример простой модели апериодической системы первого порядка
    tau = 1.0  # Временная постоянная
    K = 1.0    # Коэффициент усиления
    y = K * control * (1 - np.exp(-dt / tau))
    return y

# Параметры симуляции
setpoint = 100
dt = 0.1
time_steps = 1000

# Различные значения коэффициентов ПИД
Kp_values = [1.0, 2.0, 3.0]
Ki_values = [0.1, 0.2, 0.3]
Kd_values = [0.0, 0.1, 0.2]

# Симуляция и визуализация
plt.figure(figsize=(12, 8))

for Kp in Kp_values:
    pid = PIDController(Kp=Kp, Ki=0.1, Kd=0.0)
    measurement = 0
    measurements = []

    for _ in range(time_steps):
        control = pid.update(setpoint, measurement, dt)
        measurement = aperiodic_system_response(control, dt)
        measurements.append(measurement)

    plt.plot(measurements, label=f'Kp={Kp}')

plt.title('Influence of Kp on Transient Response (Aperiodic System)')
plt.xlabel('Time')
plt.ylabel('Measurement')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))

for Ki in Ki_values:
    pid = PIDController(Kp=1.0, Ki=Ki, Kd=0.0)
    measurement = 0
    measurements = []

    for _ in range(time_steps):
        control = pid.update(setpoint, measurement, dt)
        measurement = aperiodic_system_response(control, dt)
        measurements.append(measurement)

    plt.plot(measurements, label=f'Ki={Ki}')

plt.title('Influence of Ki on Transient Response (Aperiodic System)')
plt.xlabel('Time')
plt.ylabel('Measurement')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))

for Kd in Kd_values:
    pid = PIDController(Kp=1.0, Ki=0.1, Kd=Kd)
    measurement = 0
    measurements = []

    for _ in range(time_steps):
        control = pid.update(setpoint, measurement, dt)
        measurement = aperiodic_system_response(control, dt)
        measurements.append(measurement)

    plt.plot(measurements, label=f'Kd={Kd}')

plt.title('Influence of Kd on Transient Response (Aperiodic System)')
plt.xlabel('Time')
plt.ylabel('Measurement')
plt.legend()
plt.show()

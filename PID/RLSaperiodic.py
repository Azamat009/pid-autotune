import numpy as np
import matplotlib.pyplot as plt

class RLSPID:
    def __init__(self, Kp, Ki, Kd, lambda_):
        self.theta = np.array([Kp, Ki, Kd])
        self.P = np.eye(3) / lambda_
        self.lambda_ = lambda_
        self.phi = np.zeros(3)
        self.e_integral = 0
        self.prev_error = 0
        self.Kp_history = []
        self.Ki_history = []
        self.Kd_history = []
        self.error_history = []

    def update(self, setpoint, measurement, dt):
        error = setpoint - measurement
        self.e_integral += error * dt
        e_derivative = (error - self.prev_error) / dt
        self.prev_error = error

        self.phi = np.array([error, self.e_integral, e_derivative])
        K = np.dot(self.P, self.phi) / (self.lambda_ + np.dot(np.dot(self.phi.T, self.P), self.phi))
        self.theta += K * (measurement - np.dot(self.theta, self.phi))
        self.P = (self.P - np.dot(np.dot(K, self.phi.T), self.P)) / self.lambda_

        control = np.dot(self.theta, self.phi)

        # Сохранение истории коэффициентов и ошибки
        self.Kp_history.append(self.theta[0])
        self.Ki_history.append(self.theta[1])
        self.Kd_history.append(self.theta[2])
        self.error_history.append(error)

        return control

def aperiodic_system_response_second_order(control, dt, tau=1.0, zeta=1.0, K=1.0):
    # Пример простой модели апериодической системы второго порядка
    y = K * control * (1 - np.exp(-zeta * dt / tau) * np.cos(np.sqrt(1 - zeta**2) * dt / tau))
    return y

# Параметры симуляции
setpoint = 100
dt = 0.1
time_steps = 1000

# Инициализация ПИД-регулятора
pid = RLSPID(Kp=1.0, Ki=0.1, Kd=0.0, lambda_=0.99)
measurement = 0
measurements = []

# Симуляция
for _ in range(time_steps):
    control = pid.update(setpoint, measurement, dt)
    measurement = aperiodic_system_response_second_order(control, dt)
    measurements.append(measurement)

# Визуализация
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(pid.Kp_history, label='Kp')
plt.title('Kp History')
plt.xlabel('Time')
plt.ylabel('Kp')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(pid.Ki_history, label='Ki')
plt.title('Ki History')
plt.xlabel('Time')
plt.ylabel('Ki')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(pid.Kd_history, label='Kd')
plt.title('Kd History')
plt.xlabel('Time')
plt.ylabel('Kd')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(pid.error_history, label='Error')
plt.title('Error History')
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()

plt.tight_layout()
plt.show()

# Визуализация переходного процесса
plt.figure(figsize=(12, 6))
plt.plot(measurements, label='Measurement')
plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
plt.title('Transient Response of Aperiodic System (Second Order)')
plt.xlabel('Time')
plt.ylabel('Measurement')
plt.legend()
plt.show()

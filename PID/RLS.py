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

# Пример использования
pid = RLSPID(Kp=1.0, Ki=0.1, Kd=0.0, lambda_=0.99)
setpoint = 100
measurement = 0
dt = 0.1

for _ in range(1000):
    control = pid.update(setpoint, measurement, dt)
    # Обновление измерения на основе модели системы
    measurement = control  # Пример простой модели

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

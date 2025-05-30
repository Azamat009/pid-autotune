import numpy as np
import matplotlib.pyplot as plt

class GradientDescentPID:
    def __init__(self, Kp, Ki, Kd, alpha):
        self.theta = np.array([Kp, Ki, Kd])
        self.alpha = alpha
        self.e_integral = 0
        self.prev_error = 0
        self.Kp_history = []
        self.Ki_history = []
        self.Kd_history = []
        self.error_history = []

    def update(self, setpoint, measurement, dt):
        error = setpoint - measurement

        # Ограничение значений ошибки
        error = np.clip(error, -1e6, 1e6)

        self.e_integral += error * dt
        e_derivative = (error - self.prev_error) / dt
        self.prev_error = error

        phi = np.array([error, self.e_integral, e_derivative])
        control = np.dot(self.theta, phi)

        # Проверка на NaN и Inf
        if np.isnan(error) or np.isinf(error):
            error = 0

        # Градиент функции потерь
        gradient = -2 * error * phi

        # Проверка на NaN и Inf в градиенте
        if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            gradient = np.zeros_like(gradient)

        # Обновление параметров
        self.theta -= self.alpha * gradient

        # Сохранение истории коэффициентов и ошибки
        self.Kp_history.append(self.theta[0])
        self.Ki_history.append(self.theta[1])
        self.Kd_history.append(self.theta[2])
        self.error_history.append(error)

        return control

# Пример использования
pid = GradientDescentPID(Kp=1.0, Ki=0.1, Kd=0.0, alpha=0.01)
setpoint = 100
measurement = 0
dt = 0.1

for _ in range(1000):
    control = pid.update(setpoint, measurement, dt)
    # Обновление измерения на основе модели системы
    measurement = control  # Пример простой модели

# Визуализация
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(pid.Kp_history, label='Kp')
plt.title('Kp History')
plt.xlabel('Time')
plt.ylabel('Kp')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(pid.Ki_history, label='Ki')
plt.title('Ki History')
plt.xlabel('Time')
plt.ylabel('Ki')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(pid.Kd_history, label='Kd')
plt.title('Kd History')
plt.xlabel('Time')
plt.ylabel('Kd')
plt.legend()

plt.tight_layout()
plt.show()

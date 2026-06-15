"""
Библиотека для градиентного спуска с метрикой SMAPE.
Вариант 2: SMAPE (Symmetric Mean Absolute Percentage Error)
"""

import numpy as np


class SMAPELoss:
    """
    Функция потерь SMAPE (Symmetric Mean Absolute Percentage Error).
    
    Формула: SMAPE = 100/n * sum(|y_true - y_pred| / ((|y_true| + |y_pred|)/2))
    Часто используют: SMAPE = 100/n * sum(2 * |y_true - y_pred| / (|y_true| + |y_pred| + eps))
    """
    
    def __init__(self, eps=1e-8):
        """
        eps: малая константа для избежания деления на ноль
        """
        self.eps = eps
    
    def forward(self, y_true, y_pred):
        """
        Вычисление значения SMAPE.
        
        Args:
            y_true: истинные значения (numpy array)
            y_pred: предсказанные значения (numpy array)
        
        Returns:
            значение SMAPE (скаляр)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        numerator = 2.0 * np.abs(y_true - y_pred)
        denominator = np.abs(y_true) + np.abs(y_pred) + self.eps
        
        return np.mean(numerator / denominator) * 100.0
    
    def gradient(self, y_true, y_pred):
        """
        Вычисление градиента SMAPE по y_pred.
        
        ∂SMAPE/∂y_pred = 100/n * (2 * sign(y_pred - y_true) * (|y_true| + |y_pred| + eps)
                          - 2 * |y_true - y_pred| * sign(y_pred)) 
                          / (|y_true| + |y_pred| + eps)^2
        
        Args:
            y_true: истинные значения
            y_pred: предсказанные значения
        
        Returns:
            градиент по y_pred (numpy array)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        abs_y_true = np.abs(y_true)
        abs_y_pred = np.abs(y_pred)
        abs_diff = np.abs(y_true - y_pred)
        
        denominator = abs_y_true + abs_y_pred + self.eps
        
        # Производная числителя по y_pred: d/dy_pred (2|y_true - y_pred|) = 2 * sign(y_pred - y_true)
        d_numerator = 2.0 * np.sign(y_pred - y_true)
        
        # Производная знаменателя по y_pred: d/dy_pred (|y_true| + |y_pred|) = sign(y_pred)
        d_denominator = np.sign(y_pred)
        
        # Правило частного: d/dy_pred (num/den) = (d_num * den - num * d_den) / den^2
        grad = (d_numerator * denominator - abs_diff * d_denominator) / (denominator ** 2)
        
        return grad * 100.0 / len(y_true)


class LinearRegression:
    """
    Линейная регрессия с обучением через градиентный спуск.
    Использует SMAPE как функцию потерь.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=True):
        """
        Args:
            learning_rate: скорость обучения
            n_iterations: количество итераций градиентного спуска
            verbose: печатать ли прогресс обучения
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.loss_fn = SMAPELoss()
    
    def _predict_raw(self, X):
        """Вычисление линейной комбинации y = X*w + b"""
        return np.dot(X, self.weights) + self.bias
    
    def fit(self, X, y):
        """
        Обучение модели градиентным спуском.
        
        Args:
            X: матрица признаков (n_samples, n_features)
            y: целевая переменная (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        # Инициализация параметров
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        
        # Градиентный спуск
        for i in range(self.n_iterations):
            # Прямой проход
            y_pred = self._predict_raw(X)
            
            # Вычисление ошибки
            loss = self.loss_fn.forward(y, y_pred)
            self.loss_history.append(loss)
            
            # Вычисление градиентов
            grad_loss = self.loss_fn.gradient(y, y_pred)
            
            # Градиент по параметрам (через цепное правило)
            grad_weights = np.dot(X.T, grad_loss) / n_samples
            grad_bias = np.mean(grad_loss)
            
            # Обновление параметров
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias
            
            # Логирование
            if self.verbose and (i + 1) % max(1, self.n_iterations // 10) == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, SMAPE: {loss:.4f}%")
    
    def predict(self, X):
        """Предсказание значений для новых данных"""
        if self.weights is None:
            raise RuntimeError("Модель не обучена. Вызовите fit() сначала.")
        return self._predict_raw(np.array(X))
    
    def score(self, X, y):
        """Вычисление SMAPE на тестовых данных"""
        y_pred = self.predict(X)
        return self.loss_fn.forward(y, y_pred)


class GradientDescentOptimizer:
    """
    Общий класс для градиентного спуска с различными стратегиями.
    """
    
    def __init__(self, loss_fn, learning_rate=0.01, momentum=0.0, n_iterations=1000):
        """
        Args:
            loss_fn: объект функции потерь (должен иметь методы forward и gradient)
            learning_rate: скорость обучения
            momentum: коэффициент импульса (0 = стандартный SGD)
            n_iterations: количество итераций
        """
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_iterations = n_iterations
        
    def optimize(self, X, y, initial_params):
        """
        Оптимизация параметров градиентным спуском.
        
        Args:
            X: входные данные
            y: целевые значения
            initial_params: начальные параметры (словарь или список)
        
        Returns:
            optimized_params, loss_history
        """
        params = [np.array(p) for p in initial_params] if isinstance(initial_params, (list, tuple)) else [initial_params]
        param_names = ['param' + str(i) for i in range(len(params))]
        
        # Для momentum
        velocities = [np.zeros_like(p) for p in params]
        
        loss_history = []
        
        for i in range(self.n_iterations):
            # Здесь нужно определить функцию модели, которая принимает параметры и X
            # Для простоты - этот метод требует переопределения в наследниках
            raise NotImplementedError("Этот метод должен быть переопределен для конкретной модели")
        
        return params, loss_history


# Утилиты для работы с данными
def train_test_split(X, y, test_size=0.2, random_state=None):
    """Разделение данных на обучающую и тестовую выборки"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - test_size))
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def normalize(X):
    """Нормализация данных (Z-score)"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X - mean) / std, mean, std


# Пример использования
if __name__ == "__main__":
    # Генерация синтетических данных
    np.random.seed(42)
    X = np.random.randn(200, 1)
    y = 3 * X.squeeze() + 2 + np.random.randn(200) * 0.5
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Создание и обучение модели
    model = LinearRegression(learning_rate=0.05, n_iterations=500, verbose=True)
    model.fit(X_train, y_train)
    
    # Оценка
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nРезультаты:")
    print(f"Train SMAPE: {train_score:.4f}%")
    print(f"Test SMAPE: {test_score:.4f}%")
    
    # Визуализация обучения (если есть matplotlib)
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(model.loss_history)
        plt.xlabel('Итерация')
        plt.ylabel('SMAPE, %')
        plt.title('Кривая обучения')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.scatter(X_test, y_test, alpha=0.6, label='Тестовые данные')
        X_sorted = np.sort(X_test, axis=0)
        y_pred_sorted = model.predict(X_sorted)
        plt.plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label='Предсказания')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Результаты на тестовых данных')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass

#мяумяу

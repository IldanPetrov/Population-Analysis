import pandas as pd
import numpy as np
from numpy.ma.core import choose

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess


def advanced_window_smoothing(series, value_col, window_size = 3):
    """
    Улучшенное сглаживание с окном ±3 минуты и интерполяцией после 24.75 мин

    Параметры:
    ----------
    series : pd.Series или pd.DataFrame
        Входные данные с временным индексом в минутах
    value_col : str
        Название столбца со значениями для сглаживания

    Возвращает:
    ----------
    pd.Series с сглаженными значениями
    """
    # Извлекаем данные
    time_values = series.index.to_numpy()
    values = series[value_col].to_numpy()

    # Создаем целевую сетку времени:
    # - до 24.75 мин: исходные точки
    # - после 24.75 мин: шаг 0.25 мин
    target_times = []
    for t in time_values:
        if t <= 24.75:
            target_times.append(t)
        else:
            # Добавляем точки с шагом 0.25 после 24.75
            next_t = np.ceil(t * 4) / 4  # Округляем до ближайшего 0.25
            target_times.extend(np.arange(24.75, next_t + 0.25, 0.25))

    target_times = np.unique(target_times)  # Удаляем дубликаты

    # Функция для расчета взвешенного среднего в окне
    def weighted_mean(t):
        window = (time_values >= t - window_size) & (time_values <= t + window_size)
        window_values = values[window]
        if len(window_values) == 0:
            return np.nan
        # Гауссовы веса (больший вес ближе к центру)
        weights = np.exp(-(time_values[window] - t) ** 2 / (2 * 1.0 ** 2))
        return np.average(window_values, weights=weights)

    # Применяем сглаживание ко всем целевым точкам
    smoothed = np.array([weighted_mean(t) for t in target_times])

    # Интерполяция обратно к исходным временным точкам
    interp_func = interp1d(
        target_times, smoothed,
        kind='linear',
        bounds_error=False,
        fill_value="extrapolate"
    )

    return pd.Series(interp_func(time_values), index=series.index)


def adaptive_smoothing_indexed(series, value_col):
    """
    Адаптивное сглаживание с точным учетом временных интервалов
    """
    time_values = series.index.to_numpy()
    values = series[value_col].to_numpy()
    smoothed = np.zeros_like(values)

    for i, t in enumerate(time_values):
        if t <= 24.75:
            # Плотная область (0.25 мин интервал): окно 7 точек (~1.75 мин)
            window = 7
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed[i] = np.mean(values[start:end])
        else:
            # Разреженная область (1 мин интервал): окно 5 точек (5 мин)
            window = 5
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed[i] = np.mean(values[start:end])

    return pd.Series(smoothed, index=series.index)


def advanced_smoothing(series, value_col):
    """
    Улучшенное сглаживание с интерполяцией на точную временную сетку
    """
    time_values = series.index.to_numpy()
    values = series[value_col].to_numpy()

    # Создаем ИДЕАЛЬНУЮ сетку времени:
    # - до 24.75 мин с шагом 0.25 мин
    # - после с шагом 1 мин
    dense_time = np.arange(0, 24.75 + 0.25, 0.25)
    sparse_time = np.arange(24.75, time_values.max() + 1, 1)
    uniform_time = np.concatenate([dense_time, sparse_time])

    # Кубическая интерполяция (3-го порядка)
    interp_func = interp1d(
        time_values, values,
        kind='cubic',
        bounds_error=False,
        fill_value="extrapolate"
    )
    uniform_values = interp_func(uniform_time)

    # Раздельное сглаживание:
    dense_mask = uniform_time <= 24.75
    sparse_mask = ~dense_mask

    # Для плотной области (0-24.75 мин, шаг 0.25 мин)
    if np.any(dense_mask):
        smoothed_dense = savgol_filter(
            uniform_values[dense_mask],
            window_length=9,  # 9 точек = 2.25 мин
            polyorder=2
        )

    # Для разреженной области (>24.75 мин, шаг 1 мин)
    if np.any(sparse_mask):
        smoothed_sparse = savgol_filter(
            uniform_values[sparse_mask],
            window_length=5,  # 5 точек = 5 мин
            polyorder=2
        )

    # Объединяем результаты
    smoothed_values = np.empty_like(uniform_values)
    if np.any(dense_mask):
        smoothed_values[dense_mask] = smoothed_dense
    if np.any(sparse_mask):
        smoothed_values[sparse_mask] = smoothed_sparse

    # Возвращаем к исходным временным точкам
    final_func = interp1d(uniform_time, smoothed_values, kind='linear')
    return pd.Series(final_func(time_values), index=series.index)


def hybrid_smoothing(series, value_col):
    """
    Гибридный метод (LOESS + адаптивное сглаживание)
    """
    time_values = series.index.to_numpy()
    values = series[value_col].to_numpy()

    # Разделяем данные
    dense_mask = time_values <= 24.75
    sparse_mask = ~dense_mask

    # Для плотной области используем LOESS
    if np.any(dense_mask):
        dense_smoothed = lowess(
            values[dense_mask],
            time_values[dense_mask],
            frac=0.05,  # 5% точек в окне
            it=0,
            return_sorted=False
        )

    # Для разреженной - адаптивное среднее
    if np.any(sparse_mask):
        temp_df = pd.DataFrame({
            'time': time_values[sparse_mask],
            'value': values[sparse_mask]
        }).set_index('time')
        sparse_smoothed = adaptive_smoothing_indexed(temp_df, 'value')

    # Объединяем результаты
    smoothed_values = np.empty_like(values)
    if np.any(dense_mask):
        smoothed_values[dense_mask] = dense_smoothed
    if np.any(sparse_mask):
        smoothed_values[sparse_mask] = sparse_smoothed.values

    return pd.Series(smoothed_values, index=series.index)


def enhanced_smoothing(series, value_col):
    """
    Улучшенное сглаживание с безопасной обработкой области перехода
    """
    time_values = series.index.to_numpy()
    values = series[value_col].to_numpy()

    # 1. Создаем идеальную временную сетку
    dense_time = np.arange(0, 24.75 + 0.25, 0.25)
    sparse_time = np.arange(24.75, time_values.max() + 1, 1)
    uniform_time = np.concatenate([dense_time, sparse_time])

    # 2. Интерполяция
    interp_func = interp1d(
        time_values, values,
        kind='cubic',
        bounds_error=False,
        fill_value="extrapolate"
    )
    uniform_values = interp_func(uniform_time)

    # 3. Определяем маски для разных областей
    transition_mask = (uniform_time >= 24.0) & (uniform_time <= 26.0)
    main_mask = ~transition_mask

    # 4. Основное сглаживание
    smoothed_main = savgol_filter(
        uniform_values[main_mask],
        window_length=min(15, len(uniform_values[main_mask])),  # Безопасный размер окна
        polyorder=2
    )

    # 5. Безопасное сглаживание зоны перехода
    if np.any(transition_mask):
        transition_points = uniform_values[transition_mask]
        window = min(5, len(transition_points))  # Адаптивный размер окна

        if window >= 3:  # Минимальный размер окна для Savitzky-Golay
            smoothed_transition = savgol_filter(
                transition_points,
                window_length=window,
                polyorder=min(2, window - 1)  # Полином не больше window-1
            )
        else:
            smoothed_transition = transition_points  # Без сглаживания если мало точек

    # 6. Объединяем результаты
    smoothed_values = np.empty_like(uniform_values)
    smoothed_values[main_mask] = smoothed_main
    if np.any(transition_mask):
        smoothed_values[transition_mask] = smoothed_transition

    # 7. Гауссово сглаживание (если достаточно точек)
    if len(smoothed_values) > 5:
        from scipy.ndimage import gaussian_filter1d
        smoothed_values = gaussian_filter1d(smoothed_values, sigma=1.0)

    # 8. Возвращаем к исходным точкам
    final_func = interp1d(uniform_time, smoothed_values, kind='linear')
    result = pd.Series(final_func(time_values), index=series.index)

    # 9. Коррекция скачка (если есть точка перехода)
    if 24.75 in time_values:
        idx = np.where(time_values == 24.75)[0]
        if len(idx) > 0:
            transition_idx = idx[0]
            if transition_idx > 0 and transition_idx < len(result) - 1:
                # Плавный переход между точками до и после
                result.iloc[transition_idx] = (result.iloc[transition_idx - 1] +
                                               result.iloc[transition_idx + 1]) / 2

    return result


def interpolate_and_smooth(df, value_col, max_gap=2.0, window_length=15, polyorder=2):
    # Извлекаем время из индекса
    time_values = df.index.to_numpy()
    values = df[value_col].to_numpy()

    # Создаем равномерную сетку (адаптивно)
    min_time, max_time = time_values.min(), time_values.max()
    uniform_time = np.concatenate([
        np.arange(0, 24.75, 0.15),
        np.arange(24.75, max_time + 1, 1)
    ])

    # Интерполяция
    interp_func = interp1d(
        time_values, values,
        kind='cubic',
        bounds_error=False,
        fill_value="extrapolate"
    )
    uniform_values = interp_func(uniform_time)

    # Сглаживание
    smoothed_values = savgol_filter(
        uniform_values,
        window_length=window_length,
        polyorder=polyorder
    )

    # Возвращаем к исходным точкам времени
    final_func = interp1d(uniform_time, smoothed_values)
    return final_func(time_values)

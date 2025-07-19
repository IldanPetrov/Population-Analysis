import inspect
import os

import pandas as pd
import numpy as np
from numpy.ma.core import choose

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib.pyplot as plt

import warnings


def load_excel_data(file_path: str, sheet_name: str, columns: str | list[str]) -> pd.DataFrame:
    try:
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            usecols=columns,
            skiprows=19,
            header=0
        )

        print_df(df)
        df.dropna(axis=0, how='any', inplace=True)
        return df

    except Exception as e:
        print(f"Ошибка: {e}")
        return None


def print_df(df):
    # Ищем имя переменной в вызывающем коде
    # caller_frame = inspect.currentframe().f_back
    # for name, obj in caller_frame.f_locals.items():
    #     if obj is df:
    #         print(f"\n{name}:")
    #         print(df.head())
    #         break
    # else:
    #     print("\nUnnamed DataFrame:")
    #     print(df.head())
    pass


def find_deviations(df: pd.DataFrame, columns: list[str], threshold=0.15):
    last_values = df.loc[:, columns].iloc[-1]

    # Рассчитываем медиану
    median_val = last_values.median()

    # Вычисляем MAD (Median Absolute Deviation)
    mad = (last_values - median_val).abs().median()

    # Модифицированный Z-скор (более устойчивый)
    modified_z_score = 0.6745 * (last_values - median_val) / mad

    # Проверяем отклонения
    outliers = modified_z_score.abs() > 3  # >3 считается выбросом

    # Относительное отклонение от медианы
    relative_dev = (last_values - median_val) / median_val

    # Формируем отчет
    report = pd.DataFrame({
        'Образец': last_values.index,
        'Значение': last_values.values,
        'Отклонение от медианы (%)': relative_dev * 100,
        'Модифицированный Z-скор': modified_z_score,
        'Выброс': ['ДА' if out else 'НЕТ' for out in outliers]
    })
    report.set_index('Образец', inplace=True)
    #
    # print(report)
    # Усреднение с исключением выбросов
    valid_cols = [col for col in columns if not outliers[col]]
    invalid_cols = [col for col in columns if outliers[col]]
    if len(invalid_cols) > 0:
        print("Найдено отклонение")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(report)
        print("Чтобы удалить выберите номера через запятую или оставьте пустую строку:")
        for i, col in enumerate(invalid_cols):
            print(f"\t{i + 1}: {col}")

        # Удаление выбросов
        inp = input()
        if inp.strip() != '':
            to_pop = set(map(lambda x: int(x.strip()) - 1, inp.split(',')))
            for i in list(set(range(len(invalid_cols))) - to_pop):
                valid_cols.append(invalid_cols[i])
        else:
            valid_cols += invalid_cols

    # df['baseline_avg'] = df[valid_cols].mean(axis=1)
    return valid_cols


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


def plot_df(df, columns, x_column=None, figsize=(12, 6)):
    """
    Универсальная функция для построения графиков из DataFrame

    Параметры:
    ----------
    df : pandas.DataFrame
        DataFrame с данными
    columns : str или list
        Столбец(цы) для отображения (ось Y)
    x_column : str, optional
        Столбец для оси X (если None, используется индекс)
    figsize : tuple, optional
        Размер графика (ширина, высота)

    Примеры:
    --------
    plot_DF(df, 'temperature')  # По индексу
    plot_DF(df, ['A', 'B'], x_column='time')  # С указанием оси X
    """
    # Получаем имя переменной df из вызывающего кода
    caller_frame = inspect.currentframe().f_back
    df_name = None
    for name, obj in caller_frame.f_locals.items():
        if obj is df:
            df_name = name
            break

    # Подготовка данных
    if isinstance(columns, str):
        columns = [columns]

    x = df[x_column] if x_column is not None else df.index
    xlabel = x_column if x_column is not None else df.index.name

    # Создание графика
    plt.figure(figsize=figsize)

    # Цвета и стили линий
    line_styles = ['-', '--', '-.', ':']
    colors = plt.cm.tab10(range(len(columns)))

    # Построение графиков
    for i, col in enumerate(columns):
        plt.plot(x, df[col],
                 linestyle=line_styles[i % len(line_styles)],
                 color=colors[i],
                 label=col)

    # Оформление графика
    title = f"Columns from {df_name}" if df_name else "Data Plot"
    plt.title(title)
    plt.xlabel(xlabel if xlabel else "Index")
    plt.ylabel("Value")

    plt.grid(True, linestyle='--', alpha=0.6)

    if len(columns) > 1:
        plt.legend(loc='best')

    plt.tight_layout()
    plt.show()


def main():
    warnings.filterwarnings("ignore")
    data_sheets = os.listdir("data")
    data_sheet = [sheet for sheet in data_sheets if ".xls" in sheet]
    if len(data_sheets) > 0:
        print("Choose the data sheet you wish to load by number:")
        print('\n'.join(["\t" + str(i + 1) + ": " + x for i, x in enumerate(data_sheet)]))
    # file = data_sheet[int(input()) - 1]
    file = data_sheet[0]
    file_path = "data/" + file
    result_file = pd.ExcelWriter('results/Prepared ' + file)

    samples = load_excel_data(
        file_path,
        "Sample Setup",
        ["Well", "Well Position", "Sample Name"]
    )
    cycles = load_excel_data(
        file_path,
        "Multicomponent Data",
        ["Well", "Well Position", "Cycle", "FAM"]
    )

    sample_count = {}
    groups = {}

    def get_update_val(key):
        sample_count[key] = sample_count.get(key, 0) + 1
        groups[key[0]] = groups.get(key[0], set()) | {key}
        return sample_count[key]

    samples['Nick'] = [name + ' (' + str(get_update_val(name)) + ')' for name in samples['Sample Name']]
    samples.set_index('Nick', inplace=True)
    print_df(samples)

    # print(groups)

    merged = pd.merge(
        cycles,
        samples.reset_index(),  # Превращаем 'Nick' в столбец
        on='Well',  # Или 'Well Position'
        how='left'
    )
    print_df(merged)

    # Шаг 2: Создаём сводную таблицу
    result = merged.pivot(
        index='Cycle',
        columns='Nick',
        values='FAM'
    )
    print_df(result)

    # Применяем новую сортировку
    fine_sheet = result[samples.index.tolist()]
    print_df(fine_sheet)
    fine_sheet['Time'] = [
        (x - 1) * 0.25 if x <= 100 else 24.75 + x - 100
        for x in fine_sheet.index
    ]
    fine_sheet.set_index('Time', inplace=True)
    print_df(fine_sheet)

    fine_sheet.to_excel(result_file, sheet_name='Iterations Data', index=True, header=True)

    mean_samples = pd.DataFrame(index=fine_sheet.index, columns=sample_count.keys())

    # Выборка данных
    find_group = lambda tag: samples[samples['Sample Name'] == tag].index.tolist()

    def calc_group_mean(tag: str):
        base_tag = tag
        members = find_group(tag)

        # print(base_tag, members)
        valid_members = find_deviations(fine_sheet, members, threshold=0.15)
        # print(valid_members)

        mean_samples[tag] = fine_sheet[valid_members].mean(axis=1)
        fine_sheet.drop(members, axis=1, inplace=True)

    for group_sign in groups.keys():
        base_tag = group_sign + '-'
        groups[group_sign].remove(base_tag)

        calc_group_mean(base_tag)

        for tag in groups[group_sign]:
            members = find_group(tag)
            for col in members:
                fine_sheet[col] -= mean_samples[base_tag]

        mean_samples.drop(base_tag, axis=1, inplace=True)

    print_df(fine_sheet)
    print_df(mean_samples)
    fine_sheet.to_excel(result_file, sheet_name='Subtracted base', index=True, header=True)

    for col in fine_sheet.columns:
        fine_sheet[col] -= fine_sheet[col].min()

    fine_sheet.to_excel(result_file, sheet_name='Shifted to zero', index=True, header=True)

    for group_sign in groups.keys():
        for tag in groups[group_sign]:
            calc_group_mean(tag)

    # mean_samples.to_excel(result_file, sheet_name='Means', index=True, header=True)

    for col in mean_samples.columns:
        mean_samples[col + '_s'] = mean_samples[col]
        mean_samples[col + '_s'] = hybrid_smoothing(mean_samples, col + '_s')
        mean_samples[col + '_s'] = enhanced_smoothing(mean_samples, col + '_s')
        mean_samples[col + '_s'] = advanced_smoothing(mean_samples, col + '_s')
        # mean_samples[col + '_s'] = hybrid_smoothing(mean_samples, col + '_s')
        # mean_samples[col + '_s'] = enhanced_smoothing(mean_samples, col + '_s')
        mean_samples[col + '_s'] = advanced_smoothing(mean_samples, col + '_s')

    mean_samples.to_excel(result_file, sheet_name='Means | Smoothed', index=True, header=True)
    # mean_samples.to_excel(result_file, sheet_name='Smoothed', index=True, header=True)
    plot_df(mean_samples, mean_samples.columns)

    result_file.close()


if __name__ == '__main__':
    main()

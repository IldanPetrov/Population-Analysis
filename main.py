import inspect
import os

import pandas as pd

import matplotlib.pyplot as plt

import warnings

from smoothings import hybrid_smoothing, enhanced_smoothing, advanced_smoothing, advanced_window_smoothing
from random import randint


def load_excel_data(file_path: str, sheet_name: str, columns: str | list[str]) -> pd.DataFrame:
    # Преобразуем columns в список, если передана строка
    if isinstance(columns, str):
        columns = [columns]

    # Сначала считываем первые 50 строк без заголовка для анализа
    raw_data = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,
        nrows=50
    )

    # Ищем строку, где есть все нужные колонки
    header_row = None
    for i in range(len(raw_data)):
        # Получаем значения в строке, исключая NaN
        row_values = [str(val).strip() for val in raw_data.iloc[i] if pd.notna(val)]

        # Проверяем, есть ли все искомые колонки
        if all(col in row_values for col in columns):
            header_row = i
            break

    if header_row is None:
        print(f"Ошибка: не найдены все указанные колонки: {columns}")
        return None

    # print(header_row)
    # Теперь правильно считываем данные
    try:
        df = pd.read_excel(
            file_path,
            usecols=columns,
            sheet_name=sheet_name,
            skiprows=header_row,
            header=0,
            engine='openpyxl'  # Явно указываем движок для .xlsx файлов
        )

        df.dropna(axis=0, how='any', inplace=True)

        return df

    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None


def print_df(df):
    # Ищем имя переменной в вызывающем коде
    caller_frame = inspect.currentframe().f_back
    for name, obj in caller_frame.f_locals.items():
        if obj is df:
            print(f"\n{name}:")
            print(df.head())
            break
    else:
        print("\nUnnamed DataFrame:")
        print(df.head())


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
        print("Найдено отклонение!!! " * 3)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(report)

        reason = ['', ' для пощады', ', чтобы помиловать', ', чтобы не заморачиваться'][randint(0, 3)]
        print(f"Чтобы удалить выберите номера через запятую или оставьте пустую строку{reason}:")
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


def plot_df(df, columns=None, x_column=None, figsize=(12, 6)):
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
    if columns is None:
        columns = df.columns
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
    data_sheets = [sheet for sheet in data_sheets if ".xlsx" in sheet]
    # print(data_sheets)
    if len(data_sheets) > 0:
        print("Choose the data sheet you wish to load by number:")
        print('\n'.join(["\t" + str(i + 1) + ": " + x for i, x in enumerate(data_sheets)]))
    file = data_sheets[int(input()) - 1]
    # file = data_sheets[0]
    file_path = "data/" + file
    result_file = pd.ExcelWriter('results/Prepared ' + file)

    samples = load_excel_data(
        file_path,
        "Sample Setup",
        ["Well Position", "Sample Name"]
    )
    cycles = load_excel_data(
        file_path,
        "Multicomponent Data",
        ["Well Position", "Cycle", "FAM"]
    )

    sample_count = {}
    groups = {}

    def get_update_val(key):
        sample_count[key] = sample_count.get(key, 0) + 1
        groups[key[0]] = groups.get(key[0], set()) | {key}
        return sample_count[key]

    samples['Nick'] = [name + ' (' + str(get_update_val(name)) + ')' for name in samples['Sample Name']]
    samples.set_index('Nick', inplace=True)

    samples['Well Position'] = samples['Well Position'].astype(str).str.strip().str.upper()
    cycles['Well Position'] = cycles['Well Position'].astype(str).str.strip().str.upper()

    print_df(cycles)
    print_df(samples)
    print(sorted(cycles['Well Position'].unique()))
    print(sorted(samples['Well Position'].unique()))

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    #     print(samples)
    #     print(cycles)

    merged = pd.merge(
        cycles,
        samples.reset_index(),  # Превращаем 'Nick' в столбец
        on='Well Position',
        how='left'
    )

    dupes = merged[merged.duplicated(subset=['Cycle', 'Nick'], keep=False)]
    print(dupes.sort_values(['Nick', 'Cycle']))

    # Шаг 2: Создаём сводную таблицу
    result = merged.pivot(
        index='Cycle',
        columns='Nick',
        values='FAM'
    )

    # Применяем новую сортировку
    fine_sheet = result[samples.index.tolist()]
    fine_sheet['Time'] = [
        (x - 1) * 0.25 if x <= 100 else 24.75 + x - 100
        for x in fine_sheet.index
    ]
    fine_sheet.set_index('Time', inplace=True)

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

    fine_sheet.to_excel(result_file, sheet_name='Subtracted base', index=True, header=True)

    for col in fine_sheet.columns:
        fine_sheet[col] -= fine_sheet[col].min()

    fine_sheet.to_excel(result_file, sheet_name='Shifted to zero', index=True, header=True)

    for group_sign in groups.keys():
        for tag in groups[group_sign]:
            calc_group_mean(tag)

    mean_samples.to_excel(result_file, sheet_name='Means', index=True, header=True)
    plot_df(mean_samples)
    smoothed = pd.DataFrame(index=mean_samples.index, columns=mean_samples.columns)

    for col in mean_samples.columns:
        # mean_samples[col + '_s'] = mean_samples[col]
        smoothed[col] = advanced_window_smoothing(mean_samples, col, window_size=1)
        # mean_samples[col + '_s'] = hybrid_smoothing(mean_samples, col + '_s')
        # mean_samples[col + '_s'] = enhanced_smoothing(mean_samples, col + '_s')
        smoothed[col] = advanced_smoothing(smoothed, col)
        # mean_samples[col + '_s'] = advanced_smoothing(mean_samples, col + '_s')

    # mean_samples.to_excel(result_file, sheet_name='Means | Smoothed', index=True, header=True)
    # smoothed.to_excel(result_file, sheet_name='Smoothed', index=True, header=True)
    plot_df(smoothed)

    lines = smoothed.copy()
    time_range = int(input('Введите длину временного интервала для определения максимума (целое число), \n'
                           'например 4 минут: будет брать интервал из 5 значений: ')) + 1

    metrics = pd.DataFrame(index=smoothed.columns,
                           columns=['Max', 'Slope', 'Slope time', 'B', 'Time to max', 'Practice value'])
    max_val = {}
    for group_sign in groups.keys():
        max_val[group_sign] = 0
        for tag in groups[group_sign]:
            metrics.loc[tag, 'Slope'] = 0
            for i in range(smoothed.index.size, time_range - 1, -1):
                mean_val = smoothed[tag].iloc[i - time_range:i].mean()
                if mean_val > max_val[group_sign]:
                    max_val[group_sign] = mean_val

                slope = (smoothed[tag].iloc[i - 1] - smoothed[tag].iloc[i - 2]) / (
                        smoothed.index[i - 1] - smoothed.index[i - 2])
                if slope > metrics.loc[tag, 'Slope']:
                    metrics.loc[tag, 'Slope'] = slope
                    metrics.loc[tag, 'Slope time'] = smoothed.index[i - 1]
                    metrics.loc[tag, 'B'] = smoothed[tag].iloc[i - 1] - slope * smoothed.index[i - 1]
            # lines[tag + '_tang'] = metrics.loc[tag, 'B'] + metrics.loc[tag, 'Slope'] * lines.index

        metrics.loc[list(groups[group_sign]), 'Max'] = max_val[group_sign]
        lines[group_sign + '_max'] = max_val[group_sign]

    # print(max_val)
    plot_df(lines)

    lines.to_excel(result_file, sheet_name='Smoothed', index=True, header=True)

    metrics['Time to max'] = metrics['Max'] / metrics['Slope']
    metrics['Practice value'] = 3 / metrics['Time to max']

    metrics.to_excel(result_file, sheet_name='Metrics', index=True, header=True)

    result_file.close()


if __name__ == '__main__':
    main()

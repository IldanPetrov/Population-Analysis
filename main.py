import inspect
import os

import pandas as pd

from plot_service import PlotServer

import warnings

from smoothings import hybrid_smoothing, enhanced_smoothing, advanced_smoothing, advanced_window_smoothing
from random import randint


def load_excel_data(file_path: str, sheet_name: str, columns: str | list[str]) -> pd.DataFrame:
    if isinstance(columns, str):
        columns = [columns]

    # Определим альтернативные названия для некоторых колонок
    alt_names = {
        "FAM": ["SYBR"]
    }

    raw_data = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,
        nrows=50
    )

    header_row = None
    selected_columns = columns.copy()

    for i in range(len(raw_data)):
        row_values = [str(val).strip() for val in raw_data.iloc[i] if pd.notna(val)]

        test_columns = selected_columns.copy()

        # Проверяем наличие альтернатив для FAM
        for j, col in enumerate(test_columns):
            if col in alt_names and col not in row_values:
                for alt in alt_names[col]:
                    if alt in row_values:
                        test_columns[j] = alt
                        break

        if all(col in row_values for col in test_columns):
            header_row = i
            selected_columns = test_columns
            break

    if header_row is None:
        print(f"❌ Ошибка: не найдены все указанные колонки: {columns}")
        return None

    try:
        df = pd.read_excel(
            file_path,
            usecols=selected_columns,
            sheet_name=sheet_name,
            skiprows=header_row,
            header=0,
            engine='openpyxl'
        )

        df.dropna(subset=selected_columns, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Приводим названия столбцов к исходным (если заменяли FAM → SYBR)
        rename_map = {
            alt: orig
            for orig, alts in alt_names.items()
            for alt in alts
            if alt in df.columns and orig not in df.columns
        }
        df.rename(columns=rename_map, inplace=True)

        return df

    except Exception as e:
        print(f"❌ Ошибка при чтении файла '{file_path}': {e}")
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

    plotter.plot_df(df[columns], title=columns[0][:2])

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

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(report)

    reason = ['', ' для пощады', ', чтобы помиловать', ', чтобы не заморачиваться'][randint(0, 3)]
    print(f"Чтобы удалить выберите номера через запятую или оставьте пустую строку{reason}:")
    for i, col in enumerate(columns):
        print(f"\t{i + 1}: {col}")

    # Удаление выбросов
    inp = input()
    if inp.strip() != '':
        to_pop = set(map(lambda x: int(x.strip()) - 1, inp.split(',')))
        valid_cols = [col for i, col in enumerate(columns) if i not in to_pop]
    else:
        valid_cols = columns.copy()

    return valid_cols


def main():
    global plotter
    warnings.filterwarnings("ignore")
    plotter = PlotServer()

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

    merged = pd.merge(
        cycles,
        samples.reset_index(),  # Превращаем 'Nick' в столбец
        on='Well Position',
        how='left'
    )

    dupes = merged[merged.duplicated(subset=['Cycle', 'Nick'], keep=False)]

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

        valid_members = find_deviations(fine_sheet, members, threshold=0.15)
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
    plotter.plot_df(mean_samples, title='Mean')
    smoothed = pd.DataFrame(index=mean_samples.index, columns=mean_samples.columns)

    for col in mean_samples.columns:
        smoothed[col] = advanced_window_smoothing(mean_samples, col, window_size=1)
        smoothed[col] = advanced_smoothing(smoothed, col)
    plotter.plot_df(smoothed, title='Smoothed')

    lines = smoothed.copy()
    metrics = pd.DataFrame(index=smoothed.columns,
                           columns=['Max', 'Slope', 'Slope time', 'B', 'Time to max', 'Practice value'])
    # time_range = int(input('Введите длину временного интервала для определения максимума (целое число), \n'
    #                        'например 4 минут: будет брать интервал из 5 значений: ')) + 1
    time_range = 4
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

            lines[tag + '_tang'] = metrics.loc[tag, 'B'] + metrics.loc[tag, 'Slope'] * lines.index

        metrics.loc[list(groups[group_sign]), 'Max'] = max_val[group_sign]
        # lines[group_sign + '_max'] = max_val[group_sign]

    # print(max_val)
    plotter.plot_df(lines, title='Max bounds')

    lines.to_excel(result_file, sheet_name='Smoothed', index=True, header=True)

    metrics['Time to max'] = metrics['Max'] / metrics['Slope']
    metrics['Practice value'] = 3 / metrics['Time to max']

    metrics.to_excel(result_file, sheet_name='Metrics', index=True, header=True)

    result_file.close()
    plotter.close()


if __name__ == '__main__':
    main()

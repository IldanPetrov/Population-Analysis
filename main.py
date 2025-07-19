import inspect
import os

import pandas as pd
import numpy as np
from numpy.ma.core import choose


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
        print("Найдено отклонение")
        print(report)

        print("Чтобы удалить выберите номера через запятую или оставьте пустую строку:")
        for i, col in enumerate(invalid_cols):
            print(f"\t{i + 1}: {col}")

        to_pop = set(map(lambda x: int(x.strip()) - 1, input().split(',')))
        for i in list(set(range(len(invalid_cols))) - to_pop):
            valid_cols.append(invalid_cols[i])

    # df['baseline_avg'] = df[valid_cols].mean(axis=1)
    return valid_cols


def main():
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

    print(groups)

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
        members = find_group(tag)

        print(base_tag, members)
        valid_members = find_deviations(fine_sheet, members, threshold=0.15)
        print(valid_members)

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

    result_file.close()


if __name__ == '__main__':
    main()

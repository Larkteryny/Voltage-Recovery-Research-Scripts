import struct
import re
import pandas
from matplotlib import pyplot as plt
import numpy as np

def nox_extractor(INPUT_FILE):
    with open(INPUT_FILE, 'rb') as f:
        contents = f.read()

        data = []
        # AD0000E000000006
        matches = list(re.finditer(b"[^\x00][\x00\x01]\x00[\x00\x3A][^\x00]\x00\x00\x06", contents)) + list(
            re.finditer(b"[^\x00]\x00\x00[\xE0-\xFF]\x00\x00\x00\x06", contents))
        for match in matches:
            start = match.start()

            end = 0
            next_match = re.match(b'[^\x00]\x00\x00\x00[^\x00]\x00\x00\x06', contents[start + 8:])

            zero_count = 1
            begin = contents.find(b'\x00' * 150, start + 8)
            while contents[begin + zero_count * 8:begin + zero_count * 8 + 8] == b'\x00' * 8:
                zero_count += 1

            end = min(start + 8 + (next_match.start() if next_match else float('inf')),
                      contents.find(b'\x00' * 150, start + 8))
            data.append([struct.unpack('d', doublebytes)[0] for doublebytes in
                         [contents[j:j + 8] for j in range(start + 8, end, 8)]])

        print([len(d) for d in data])
        print(len(data))

        # Find mode array which is Potential Applied (V) data
        data_discretized = [tuple([abs(y) for y in x[:20]]) for x in data]
        from collections import Counter
        mode = Counter(data_discretized).most_common(1)[0]
        print(mode)
        data = [x for x in data if tuple([abs(y) for y in x[:20]]) != mode[0]]

        # [Record Signals(time, current, pxv, wev), OCP10Hz(time, wev, pxv, current), {OCP1Hz}] * 3, [LSV(Voltage applied)] * 3, [LSV(time, current, potential, voltage)]
        # Group into trials and sort
        data = [data[i:i + 4] for i in range(0, len(data), 4)]
        data.sort(key=lambda x: x[0][0])
        for i in range(len(data) - 1, 0, -1):
            if data[i][0][0] == data[i - 1][0][0]:
                data.pop(i)  # remove duplicates
            elif len(data[i - 1][0]) == 205:
                print(data[i - 1][0][0])
        data = [sublist for matrix in data for sublist in matrix]
        print([len(d) for d in data])
        print(len(data))
        return data

if __name__ == "__main__":
    points = {
        0.5: r"C:\Users\user\Downloads\2025_summer_research_data\raw data\2025_08_04\Alternating Automated Voltage Recovery Whole Cell 0.5V with 50s 0V2hrshort 8.03 overnight.nox",
        1: r"C:\Users\user\Downloads\2025_summer_research_data\raw data\2025_08_07\Alternating Automated Voltage Recovery Whole Cell 1V with 50s 0V2hrshort 8.06 overnight.nox",
        1.5: r"C:\Users\user\Downloads\2025_summer_research_data\raw data\2025_08_05\Alternating Automated Voltage Recovery Whole Cell 1.5V with 50s 0V2hrshort 8.04 overnight.nox",
        2: r"C:\Users\user\Downloads\2025_summer_research_data\raw data\2025_08_06\Alternating Automated Voltage Recovery Whole Cell 2V with 50s 0V2hrshort 8.05 overnight.nox",
    }

    discharged = []
    for held in sorted(points.keys()):
        import csv
        with open(rf"C:\Users\user\Downloads\2025_summer_research_data_q2temp\{held}.csv", 'r') as f:
            data = csv.reader(f)
            time = []
            cur = []
            for i, row in enumerate(data):
                if i == 0: continue
                time.append(float(row[1]))
                cur.append(float(row[2]))

            cur_integral = 0
            for i in range(len(time) - 1):
                cur_integral += ((cur[i] + cur[i + 1]) / 2) * (time[i + 1] - time[i])
            discharged.append(abs(cur_integral))

    print(discharged)

    points_list = sorted(points.keys())

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.errorbar(points_list, discharged, fmt='o', capsize=5)
    plt.xlabel('Charging Voltage (V)')
    plt.ylabel('Discharged (C)')
    plt.title('Total Discharged Charge')
    plt.grid(True)
    plt.xlim(left=0)
    #plt.ylim(bottom=0)
    plt.axvline(x=1.23, color='k', linestyle=':')

    plt.subplot(1, 3, 2)
    plt.errorbar(points_list[:3], discharged[:3], fmt='o', capsize=5)
    plt.xlabel('Charging Voltage (V)')
    plt.ylabel('Discharged (C)')
    plt.title('Total Discharged Charge (excl. 2V)')
    plt.grid(True)
    plt.xlim(left=0)

    # Linear fit
    x_fit = np.array(points_list[:-1])
    y_fit = np.array(discharged[:-1])
    coeffs = (np.polynomial.Polynomial.fit(x_fit, y_fit, 1)).convert().coef
    x_smooth = np.linspace(0, max(points_list), 40)
    y_smooth = coeffs[1] * x_smooth + coeffs[0]
    # R^2 for linear fit
    y_pred = coeffs[1] * x_fit + coeffs[0]
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r2_lin = 1 - (ss_res / ss_tot)
    plt.plot(x_smooth, y_smooth, 'r--', label=f'{1000*coeffs[1]:.2f} x + {1000*coeffs[0]:.2f} (mC)    $R^2$={r2_lin:.3f}')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.errorbar(points_list[:3], [discharged[i] / points_list[i] for i in range(3)], fmt='o', capsize=5)
    plt.xlabel('Charging Voltage (V)')
    plt.ylabel('Discharged per Volt (C/V)')
    plt.title('Total Discharged Charge per Volt Held (excl. 2V)')
    plt.grid(True)
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.subplots_adjust(wspace=0.4)

    plt.show()
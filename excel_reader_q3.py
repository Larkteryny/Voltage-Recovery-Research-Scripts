from operator import indexOf
import struct
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

def nox_extractor(INPUT_FILE):
    with open(INPUT_FILE, 'rb') as f:
        contents = f.read()

        data = []
        matches = list(re.finditer(b"[^\x00][\x00\x01]\x00[\x00\x3A][^\x00]\x00\x00\x06", contents))
        for match in matches:
            start = match.start()

            end = 0
            next_match = re.match(b'[^\x00]\x00\x00\x00[^\x00]\x00\x00\x06', contents[start + 8:])
            print(next_match.start() if next_match else -1, contents.find(b'\x00' * 200, start + 8))

            zero_count = 1
            begin = contents.find(b'\x00' * 800, start + 8)
            while contents[begin + zero_count * 8:begin + zero_count*8 + 8] == b'\x00'*8:
                zero_count += 1
            print(zero_count)

            end = min(start + 8 + (next_match.start() if next_match else float('inf')), contents.find(b'\x00' * 800, start + 8))

            data.append([struct.unpack('d', doublebytes)[0] for doublebytes in [contents[j:j + 8] for j in range(start + 8, end, 8)]])

        # [Record Signals(time, current, pxv, wev), OCP10Hz(time, wev, pxv, current), {OCP1Hz}] * 3, [LSV(Voltage applied)] * 3, [LSV(time, current, potential, voltage)]
        # Group into trials and sort
        #data = data[:36] + data[39:]
        data = [data[i:i + 4] for i in range(0, len(data), 4)]
        #data.sort(key=lambda x: x[0][0])
        data = [sublist for matrix in data for sublist in matrix]
        print([len(d) for d in data])
        print(len(data))
        return data

points = {
    1: r"C:\Users\user\Downloads\2025_summer_research_data\raw data\2025_08_22\Whole Cell +1V7200s 1sdischarge 0V2hrshort 8.21 overnight.nox",
    5: r"C:\Users\user\Downloads\2025_summer_research_data\raw data\2025_08_21\Whole Cell +1V7200s 5sdischarge 0V2hrshort 8.20 overnight.nox",
    10: r"C:\Users\user\Downloads\2025_summer_research_data\raw data\2025_08_23\Whole Cell +1V7200s 10sdischarge 0V2hrshort 8.22 overnight.nox",
    20: r"C:\Users\user\Downloads\2025_summer_research_data\raw data\2025_08_24\Whole Cell +1V7200s 20sdischarge 0V2hrshort 8.23 overnight.nox",
    50: r"C:\Users\user\Downloads\2025_summer_research_data\raw data\2025_08_25\Whole Cell +1V7200s 50sdischarge 0V2hrshort 8.24 overnight.nox",
    100: r"C:\Users\user\Downloads\Whole Cell +1V7200s 100sdischarge 0V2hrshort 8.25 overnight.nox",
}
extracted = {p: [] for p in points.keys()}

for p, file in points.items():
    data = nox_extractor(file)
    for i, c in enumerate(data):
        if "i % 16 in [9, 13]" and len(c) in [1500, 300] and abs(c[-2]) > abs(c[-1]) and abs(c[1]) > abs(c[0]) and abs(data[i - 1][-2]) < abs(data[i - 1][-1]):  # one of 2 ocps, and contains a maxima
            if len(c) == 300:
                extracted[p].append([data[i - 1], c])  # Time (s), WE(1).Potential (V)
                extracted[p][-1][0][0] -= 150
            else:
                extracted[p].append([data[i - 1], c])  # Time (s), WE(1).Potential (V)
            #plt.plot(data[i - 1], c)
            #plt.show()

processed = {p: [] for p in points.keys()}
for p, data in extracted.items():
    rebounds = []  # negatives, positives
    times = []
    for trial in data:
        rebounds.append(max(trial[1]))
        times.append(trial[0][indexOf(trial[1], rebounds[-1])] - trial[0][0])

    rebounds_stddevs = np.std(rebounds)
    times_stddevs = np.std(times)

    processed[p] = [rebounds, rebounds_stddevs, times, times_stddevs]
    print(times)

results = {p: [] for p in points.keys()}
for p, data in processed.items():
    results[p].append(np.average([abs(np.average(data[0][:3]))]))  # mean rebound
    results[p].append(data[1] / results[p][-1])  # coeff of variation
    results[p].append(np.average([abs(np.average(data[2][:3]))]))  # mean time of rebounds
    results[p].append(data[3] / results[p][-1])  # coeff of variation

for p, r in results.items():
    print(p, *r, sep="\t")

points_list = sorted(points.keys())
mean_rebounds = [results[p][0] for p in points_list]
rebound_err = [results[p][1] * results[p][0] for p in points_list]
mean_times = [results[p][2] for p in points_list]
time_err = [results[p][3] * results[p][2] for p in points_list]

plt.figure(figsize=(10, 4))

# Mean rebound
plt.subplot(1, 2, 1)
plt.errorbar(points_list, mean_rebounds, yerr=rebound_err, fmt='o', capsize=5)
plt.xlabel('Holding Time (s)')
plt.ylabel('Rebound Voltage (V)')
plt.title('Mean Rebound Voltage')
plt.grid(True)
plt.xlim(left=0)
plt.ylim(bottom=0)

# Logarithmic fit
x_fit = np.array(points_list)
y_fit = np.array(mean_rebounds)
log_x = np.log(x_fit)
coeffs = np.polyfit(log_x, y_fit, 1)
x_smooth = np.geomspace(min(points_list), max(points_list), 40)
y_logfit_smooth = coeffs[0] * np.log(x_smooth) + coeffs[1]
# R^2 for log fit
y_log_pred = coeffs[0] * log_x + coeffs[1]
ss_res_log = np.sum((y_fit - y_log_pred) ** 2)
ss_tot_log = np.sum((y_fit - np.mean(y_fit)) ** 2)
r2_log = 1 - (ss_res_log / ss_tot_log)
plt.plot(x_smooth, y_logfit_smooth, 'r--', label=f'{coeffs[0]:.2f} log(x) + {coeffs[1]:.2f}    $R^2$={r2_log:.3f}')

# Exponential fit
def exp_func(x, a, b, c):
    try:
        with np.errstate(over='raise'):
            result = -a * np.exp(-b * x) + c
    except FloatingPointError:
        # If overflow, return a large value (same shape as x)
        result = np.full_like(x, fill_value=1e6, dtype=np.float64)
    # Also handle any inf values that sneak through
    result = np.where(np.isfinite(result), result, 1e6)
    return result
x_fit = np.array(points_list) / 10000
y_fit = np.array(mean_rebounds)
popt, _ = curve_fit(exp_func, x_fit, y_fit, p0=[1, 1, 0.5], maxfev=10000)
print(popt)
x_smooth = np.geomspace(min(points_list)/10000, max(points_list)/10000, 40)
y_expfit_smooth = exp_func(x_smooth, *popt)
# R^2 for exp fit
y_exp_pred = exp_func(x_fit, *popt)
ss_res_exp = np.sum((y_fit - y_exp_pred) ** 2)
ss_tot_exp = np.sum((y_fit - np.mean(y_fit)) ** 2)
r2_exp = 1 - (ss_res_exp / ss_tot_exp)
plt.plot(
    x_smooth * 10000,
    y_expfit_smooth,
    'g--',
    label=f'-{popt[0]:.2f} exp({(-popt[1]/10000):.2f} x) + {popt[2]:.2f}    $R^2$={r2_exp:.3f}'
)
plt.legend()

# Duplicate of first graph with log x axis
plt.subplot(1, 2, 2)
plt.errorbar(points_list, mean_rebounds, yerr=rebound_err, fmt='o', capsize=5)
plt.xlabel('Holding Time (s) [log]')
plt.ylabel('Rebound Voltage (V)')
plt.title('Mean Rebound Voltage (Log X)')
plt.grid(True, which='both')
plt.xscale('log')
plt.ylim(bottom=0)
plt.plot(x_smooth*10000, y_logfit_smooth, 'r--', label=f'{coeffs[0]:.2f} log(x) + {coeffs[1]:.2f}    $R^2$={r2_log:.3f}')
plt.plot(
    x_smooth*10000,
    y_expfit_smooth,
    'g--',
    label=f'-{popt[0]:.2f} exp(-{(-popt[1]/10000):.2f} x) + {popt[2]:.2f}    $R^2$={r2_exp:.3f}'
)
plt.legend()

plt.show()

# Mean time of rebounds
plt.subplot(1, 2, 1)
plt.errorbar(points_list, mean_times, yerr=time_err, fmt='o', capsize=5)
plt.xlabel('Holding Time (s)')
plt.ylabel('Time of Peak Rebound (s)')
plt.title('Mean Time of Peak Rebound')
plt.grid(True)
plt.legend()
#plt.ylim(bottom=0)
plt.xscale('log')
plt.yscale('log')

# Log fit
x_fit = np.array(points_list)
y_fit = np.array(mean_times)
log_x = np.log(x_fit)
coeffs = np.polyfit(log_x, y_fit, 1)
x_smooth = np.geomspace(min(points_list), max(points_list), 40)
y_logfit_smooth = coeffs[0] * np.log(x_smooth) + coeffs[1]
# R^2 for log fit
y_log_pred = coeffs[0] * log_x + coeffs[1]
ss_res_log = np.sum((y_fit - y_log_pred) ** 2)
ss_tot_log = np.sum((y_fit - np.mean(y_fit)) ** 2)
r2_log = 1 - (ss_res_log / ss_tot_log)
plt.plot(x_smooth, y_logfit_smooth, 'r--', label=f'{coeffs[0]:.2f} log(x) + {coeffs[1]:.2f}    $R^2$={r2_log:.3f}')
plt.legend()
plt.tight_layout()
plt.show()
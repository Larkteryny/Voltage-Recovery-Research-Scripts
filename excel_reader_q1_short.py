from operator import indexOf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

FILEPATH = r"C:\Users\user\Downloads\Voltage Recovery Project - Charge Holding Time.xlsx"
df_dict = pd.read_excel(FILEPATH, sheet_name=None)
print(df_dict['Alternating Auto1V 10s 7.12over'].keys())
#sys.exit()

points = {
    10: "7.25",
    30: "7.27",
    50: "7.29",
    100: "7.24",
    500: "7.26"
}
extracted = {p: [] for p in points.keys()}

for p, string in points.items():
    for k in df_dict.keys():
        if string in k:
            data = df_dict[k]
            cols = data.keys()
            for i, c in enumerate(cols):
                if str(c).startswith("WE(1).Potential (V)") and abs(data[c][1]) > abs(data[c][0]):  # not a long run self-discharge OCP either
                    extracted[p].append([data[cols[i - 1]], data[c]])  # Time (s), WE(1).Potential (V)
            extracted[p].sort(key=lambda x: x[0][0])
            break

processed = {p: [] for p in points.keys()}
for p, data in extracted.items():
    rebounds = [[], []]  # negatives, positives
    times = [[], []]
    for trial in data:
        if trial[1][50] < 0:
            rebounds[0].append(min(trial[1]))
            times[0].append(trial[0][indexOf(trial[1], rebounds[0][-1])] - trial[0][0])
        else:
            rebounds[1].append(max(trial[1]))
            times[1].append(trial[0][indexOf(trial[1], rebounds[1][-1])] - trial[0][0])

    rebounds_stddevs = [np.var(rebounds[0]), np.std(rebounds[1])]
    times_stddevs = [np.std(times[0]), np.std(times[1])]

    processed[p] = [rebounds, rebounds_stddevs, times, times_stddevs]
    print(times)

results = {p: [] for p in points.keys()}
for p, data in processed.items():
    results[p].append(np.average([abs(np.average(data[0][0][:3])), abs(np.average(data[0][1][:3]))]))  # mean rebound
    results[p].append(max(data[1]) / results[p][-1])  # coeff of variation
    results[p].append(np.average([abs(np.average(data[2][0][:3])), abs(np.average(data[2][1][:3]))]))  # mean time of rebounds
    results[p].append(max(data[3]) / results[p][-1])  # coeff of variation

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
x_smooth = np.linspace(min(points_list), max(points_list), 40)
y_logfit_smooth = coeffs[0] * np.log(x_smooth) + coeffs[1]
# R^2 for log fit
y_log_pred = coeffs[0] * log_x + coeffs[1]
ss_res_log = np.sum((y_fit - y_log_pred) ** 2)
ss_tot_log = np.sum((y_fit - np.mean(y_fit)) ** 2)
r2_log = 1 - (ss_res_log / ss_tot_log)
plt.plot(x_smooth, y_logfit_smooth, 'r--', label=f'{coeffs[0]:.2f} log(x) + {coeffs[1]:.2f}    $R^2$={r2_log:.3f}')

# Exponential fit
def exp_func(x, a, b, c):
    return -a * np.exp(-b * x) + c
x_fit = np.array(points_list)
y_fit = np.array(mean_rebounds)
popt, _ = curve_fit(exp_func, x_fit, y_fit, maxfev=10000)
print(popt)
x_smooth = np.linspace(min(points_list), max(points_list), 40)
y_expfit_smooth = exp_func(x_smooth, *popt)
# R^2 for exp fit
y_exp_pred = exp_func(x_fit, *popt)
ss_res_exp = np.sum((y_fit - y_exp_pred) ** 2)
ss_tot_exp = np.sum((y_fit - np.mean(y_fit)) ** 2)
r2_exp = 1 - (ss_res_exp / ss_tot_exp)
plt.plot(
    x_smooth,
    y_expfit_smooth,
    'g--',
    label=f'-{popt[0]:.2f} exp({-popt[1]:.2f} x) + {popt[2]:.2f}    $R^2$={r2_exp:.3f}'
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
plt.plot(x_smooth, y_logfit_smooth, 'r--', label=f'{coeffs[0]:.2f} log(x) + {coeffs[1]:.2f}    $R^2$={r2_log:.3f}')
plt.plot(
    x_smooth,
    y_expfit_smooth,
    'g--',
    label=f'-{popt[0]:.2f} exp(-{popt[1]:.2f} x) + {popt[2]:.2f}    $R^2$={r2_exp:.3f}'
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
plt.ylim(bottom=0)
plt.xscale('log')

plt.tight_layout()
plt.show()
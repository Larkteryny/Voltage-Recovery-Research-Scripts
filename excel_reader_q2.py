from operator import indexOf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

FILEPATH = r"C:\Users\user\Downloads\Voltage Recovery Project Preliminary Data.xlsx"
df_dict = pd.read_excel(FILEPATH, sheet_name=None)
#sys.exit()

points = {
    0.5: "8.3",
    1: "8.6",
    1.5: "8.4",
    2: "8.5",
}
extracted = {p: [] for p in points.keys()}

for p, string in points.items():
    for k in df_dict.keys():
        if string in k:
            print("YEYEYEYE")
            data = df_dict[k]
            cols = data.keys()
            for i, c in enumerate(cols):
                if str(c).lstrip().startswith("WE(1).Potential (V)") and abs(data[c][1]) > abs(data[c][0]):  # not a long run self-discharge OCP either
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
    print(rebounds)

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
hydrolysis_threshold = 1.23

plt.figure(figsize=(10, 4))

# Mean rebound
plt.subplot(1, 2, 1)
plt.errorbar(points_list, mean_rebounds, yerr=rebound_err, fmt='o', capsize=5)
plt.xlabel('Charging Voltage (V)')
plt.ylabel('Rebound Voltage (V)')
plt.title('Mean Rebound Voltage')
plt.grid(True)
plt.xlim(left=0)
plt.ylim(bottom=0)

# Linear fit
x_fit = np.array(points_list[:-1])
y_fit = np.array(mean_rebounds[:-1])
coeffs = (np.polynomial.Polynomial.fit(x_fit, y_fit, 1)).convert().coef
x_smooth = np.linspace(0, max(points_list), 40)
y_smooth = coeffs[1] * x_smooth + coeffs[0]
# R^2 for linear fit
y_pred = coeffs[1] * x_fit + coeffs[0]
ss_res = np.sum((y_fit - y_pred) ** 2)
ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
r2_lin = 1 - (ss_res / ss_tot)
plt.plot(x_smooth, y_smooth, 'r--', label=f'{coeffs[1]:.2f} x + {coeffs[0]:.2f} (excl. 2V)    $R^2$={r2_lin:.3f}')

plt.axvline(x=hydrolysis_threshold, color='k', linestyle=':')

# Quadratic fit
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c
x_fit = np.array(points_list)
y_fit = np.array(mean_rebounds)
coeffs = (np.polynomial.Polynomial.fit(x_fit, y_fit, 2)).convert().coef
x_smooth = np.linspace(0, max(points_list), 40)
y_smooth = coeffs[2] * x_smooth * x_smooth + coeffs[1] * x_smooth + coeffs[0]
# R^2 for quadratic fit
y_pred = coeffs[2] * x_fit * x_fit + coeffs[1] * x_fit + coeffs[0]
ss_res = np.sum((y_fit - y_pred) ** 2)
ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
r2_quad = 1 - (ss_res / ss_tot)
plt.plot(x_smooth, y_smooth, 'g--', label=f'{coeffs[2]:.2f} x^2 + {coeffs[1]:.2f} x + {coeffs[0]:.2f}    $R^2$={r2_quad:.3f}')
plt.axvline(x=hydrolysis_threshold, color='k', linestyle=':')
plt.legend()

# Mean time of rebounds
plt.subplot(1, 2, 2)
plt.errorbar(points_list, mean_times, yerr=time_err, fmt='o', capsize=5)
plt.xlabel('Charging Voltage (V)')
plt.ylabel('Time of Peak Rebound (s)')
plt.title('Mean Time of Peak Rebound')
plt.grid(True)
plt.ylim(bottom=0)
plt.xlim(left=0)

# Logarithmic fit
x_fit = np.array(points_list)
y_fit = np.array(mean_times)
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
plt.axvline(x=hydrolysis_threshold, color='k', linestyle=':')

# Exponential fit
def exp_func(x, a, b, c):
    return -a * np.exp(-b * x) + c
x_fit = np.array(points_list)
y_fit = np.array(mean_times)
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

plt.tight_layout()
plt.show()
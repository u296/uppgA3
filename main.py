import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/todd/Downloads/rig3_measure6.tsv", sep="\t")

t = df["time"].to_numpy()
z1 = df["z2"].to_numpy()
samplerate = 100.0

z1 /= 1000.0 # have units in meters



def fourier_transform(t, x, samplerate):
    fourier = np.fft.rfft(x)
    xfourier = np.fft.rfftfreq(len(t), 1.0/samplerate)

    return (xfourier, fourier)

def index_of_nearest(a, val):
    diff = np.abs(a - val)
    return diff.argmin()

def integrate_approx(xvals, yvals, start_f, stop_f):
    start_i = index_of_nearest(xvals, start_f)
    stop_i = index_of_nearest(xvals, stop_f)

    running_total = 0.0

    # this is to compensate for the starting section
    f0 = np.interp(start_f, xvals, yvals)
    f1 = yvals[start_i]

    running_total += 0.5 * (xvals[start_i] - start_f) * (f0 + f1)

    # --||-- ending section
    f2 = np.interp(stop_f, xvals, yvals)
    f3 = yvals[stop_i]

    running_total += 0.5 * (stop_f - xvals[stop_i]) * (f2 + f3)

    running_total += np.trapz(yvals[start_i:stop_i], xvals[start_i:stop_i])

    return running_total

plt.plot(t, z1)
plt.show()

t_cutoff_f = float(input("start time for fourier: "))
t_cutoff_i = index_of_nearest(t, t_cutoff_f)

trimmed_t = t[t_cutoff_i:]
trimmed_z1 = z1[t_cutoff_i:]

N = len(trimmed_t)
print(f"N = {N}")


freq, amplitude = fourier_transform(trimmed_t, trimmed_z1, samplerate)

amplitude = np.real(amplitude)

while True:
    fig, axs = plt.subplots(1,2)

    axs[0].plot(freq, amplitude, ".-")
    axs[0].plot(np.array([min(freq), max(freq)]), np.array([0.0, 0.0]), "--k")
    axs[0].set_xlabel("frequency [Hz]")
    axs[0].set_ylabel("amplitude [m]")

    axs[1].plot(t, z1)
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("position [m]")
    plt.show()

    xstart_f = float(input("freq band start:"))
    xstop_f = float(input("freq band stop:"))


    start_i = index_of_nearest(freq, xstart_f)
    stop_i = index_of_nearest(freq, xstop_f)

    print(f"nearest start: index = {start_i} val = {freq[start_i]}")
    print(f"nearest stop: index = {stop_i} val = {freq[stop_i]}")


    xvals = freq[start_i:stop_i]
    yvals = amplitude[start_i:stop_i]

    # find midpoint

    midpoint = np.average(xvals, weights=np.abs(yvals))

    amp = np.sum(yvals)
    absamp = np.sum(np.abs(yvals))

    print(f"frequency = {midpoint}")
    print(f"amplitude = {amp}")
    print(f"absolute amplitude = {absamp}")

    plt.plot(xvals, np.abs(yvals))
    plt.plot(xvals, yvals)
    plt.plot([midpoint, midpoint], [-max(abs(yvals)), max(abs(yvals))*1.1], ":k")

    plt.show()

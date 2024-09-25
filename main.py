import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = "data/rig1_measure3.tsv"
names = ["frame", "time", "x1", "y1", "z1", "x2", "y2", "z2"]

df = pd.read_csv(path, sep="\t", skiprows=11, names=names)

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

real_amplitude = np.real(amplitude)
imag_amplitude = np.imag(amplitude)
abs_amplitude = np.abs(amplitude)

while True:
    fig, axs = plt.subplots(1,2)

    axs[0].plot(freq, real_amplitude, ".-", label="real amplitude (cos alignment)")
    axs[0].plot(freq, imag_amplitude, ".-", label="imag amplitude (sin alignment)")
    axs[0].plot(freq, abs_amplitude, ".-", label="abs amplitude")
    axs[0].plot(np.array([min(freq), max(freq)]), np.array([0.0, 0.0]), "--k")
    axs[0].set_xlabel("frequency [Hz]")
    axs[0].set_ylabel("amplitude [m]")
    axs[0].legend()

    axs[1].plot(t, z1)
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("position [m]")
    fig.legend()
    plt.show()

    xstart_f = float(input("freq band start:"))
    xstop_f = float(input("freq band stop:"))


    start_i = index_of_nearest(freq, xstart_f)
    stop_i = index_of_nearest(freq, xstop_f)

    print(f"nearest start: index = {start_i} val = {freq[start_i]}")
    print(f"nearest stop: index = {stop_i} val = {freq[stop_i]}")


    freqs = freq[start_i:stop_i]
    real_amplitudes = real_amplitude[start_i:stop_i]
    imag_amplitudes = imag_amplitude[start_i:stop_i]

    # find midpoint

    real_midpoint_freq = np.average(freqs, weights=np.abs(real_amplitudes))
    imag_midpoint_freq = np.average(freqs, weights=np.abs(imag_amplitudes))

    realamp = np.sum(real_amplitudes)
    abs_realamp = np.sum(np.abs(real_amplitudes))

    imagamp = np.sum(imag_amplitudes)
    abs_imagamp = np.sum(np.abs(imag_amplitudes))

    print(f"(cos)frequency = {real_midpoint_freq}")
    print(f"(cos)real_amplitude = {realamp}")
    print(f"(cos)absolute real_amplitude = {abs_realamp}")

    print(f"(sin)frequency = {imag_midpoint_freq}")
    print(f"(sin)imag_amplitude = {imagamp}")
    print(f"(sin)absolute imag_amplitude = {abs_imagamp}")

    plt.plot(freqs, np.abs(real_amplitudes), label="absolute real amplitude")
    plt.plot(freqs, real_amplitudes, label = "real amplitude")
    plt.plot([real_midpoint_freq, real_midpoint_freq], [-max(abs(real_amplitudes)), max(abs(real_amplitudes))*1.1], ":k")

    plt.plot(freqs, np.abs(imag_amplitudes), label="absolute imag amplitude")
    plt.plot(freqs, imag_amplitudes, label = "imag amplitude")
    plt.plot([imag_midpoint_freq, imag_midpoint_freq], [-max(abs(imag_amplitudes)), max(abs(imag_amplitudes))*1.1], ":k")

    plt.legend()

    plt.show()

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


freq, amplitude = fourier_transform(trimmed_t, trimmed_z1, samplerate)

amplitude = np.abs(amplitude) / len(trimmed_t)

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

    xstart_f = float(input("freq integration start:"))
    xstop_f = float(input("freq integration stop:"))


    start_i = index_of_nearest(freq, xstart_f)
    stop_i = index_of_nearest(freq, xstop_f)

    print(f"nearest start: index = {start_i} val = {freq[start_i]}")
    print(f"nearest stop: index = {stop_i} val = {freq[stop_i]}")


    xvals = freq[start_i:stop_i]
    yvals = amplitude[start_i:stop_i]

    # find midpoint

    midpoint = 0.5 * (xvals[0] + xvals[-1]) 



    pivot_down = xvals[0]
    pivot_up = xvals[-1]


    
        


    for _ in range(100):

        lower = integrate_approx(xvals, np.abs(yvals), xvals[0], midpoint)
        upper = integrate_approx(xvals, np.abs(yvals), midpoint, xvals[-1])

        #print(f"lower = {lower:.3f} upper = {upper:.3f}")
        #print(f"midpoint = {midpoint}")
        
        if abs(lower - upper) < 0.001 * abs(lower):
            break

        if lower < upper:
            print("+")
            pivot_down = midpoint
            midpoint = 0.5 * (midpoint + pivot_up)
        else: 
            print("-")
            pivot_up = midpoint
            midpoint = 0.5 * (midpoint + pivot_down)

        
    amp = integrate_approx(freq, amplitude, xstart_f, xstop_f)

    print(f"frequency = {midpoint}")
    print(f"amplitude = {amp}")

    plt.plot(xvals, np.abs(yvals))
    plt.plot(xvals, yvals)
    plt.plot([midpoint, midpoint], [-max(abs(yvals)), max(abs(yvals))*1.1], ":k")

    plt.show()

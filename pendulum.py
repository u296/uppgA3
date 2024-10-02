import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = "data/pendulum2.tsv"
spring_names = ["frame", "time", "x1", "y1", "z1", "x2", "y2", "z2"]
pendulum_names = ["frame", "time", "x", "y", "z"]

df = pd.read_csv(path, sep="\t", skiprows=11, names=pendulum_names)

samplerate = 100.0
t = df["time"].to_numpy()
x = df["x"]

points = df[["x", "y", "z"]].to_numpy() / 1000.0



def cleanup_pendulum(points):
    n_points = points.shape[0]
    ones = np.transpose(np.array([np.ones(n_points)]))
    zeros = np.transpose(np.zeros(n_points))
    A = np.concatenate((points, ones), axis=1)

    print("AAAAAAAAAAAAAAAAAAA")
    print(A)
    print("chatgpt solution:")
    U, S, Vt = np.linalg.svd(A)

    x_non_trivial = Vt.T[:, -1]

    x_non_trivial /= np.linalg.norm(x_non_trivial[0:3])

    normal = x_non_trivial[0:3]
    offset = x_non_trivial[3]

    print("Non-trivial solution x:", x_non_trivial)

    print("add to points: ", normal * offset)
    points_through_origin = points + normal * offset

    print("points through origin:")
    print(points_through_origin)

    print("minmax y: ")

    e1 = np.array([1.0,0.0,0.0])

    planetangent = np.cross(e1, normal)
    planecotangent = np.cross(planetangent, normal)

    print("normal", normal)
    print("tangent", planetangent)
    print("cotangent", planecotangent)

    M = np.linalg.inv(np.concatenate(([normal], [planetangent], [planecotangent]), axis=0))
    print("M", M)
    print("det M", np.linalg.det(M))

    print("points shape", points.shape)

    rotated_points = np.transpose(M @ np.transpose(points_through_origin))

    print("rotated points: ", rotated_points)


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

plt.plot(t, x)
plt.show()

t_cutoff_f = float(input("start time for min/max detection: "))
t_cutoff_i = index_of_nearest(t, t_cutoff_f)

mint = []
minx = []

maxt = []
maxx = []

laststate = 0
lastvalue = 0.0
lasttime = 0.0

if x[t_cutoff_i] < x[t_cutoff_i + 1]:
    laststate = 1
else:
    laststate = -1

for i, (time, xval) in enumerate(zip(t[t_cutoff_i:], x[t_cutoff_i:])):
    if i == 0:
        lastvalue = xval
        lasttime = time
        continue

    if lastvalue < xval:
        # we are growing
        if laststate == -1:
            # we just passed a minima
            mint.append(lasttime)
            minx.append(float(lastvalue))
        laststate = 1
    elif lastvalue > xval:
        # we are shrinking
        if laststate == 1:
            # we just passed a maxima
            maxt.append(lasttime)
            maxx.append(float(lastvalue))
        laststate = -1

    lastvalue = xval
    lasttime = time


npmint = np.array(mint)
npmaxt = np.array(maxt)

mindist = np.diff(npmint)
maxdist = np.diff(npmaxt)

np.set_printoptions(precision=3, suppress=True)

print("distances between minima")
print(mindist)

print("distances between maxima")
print(maxdist)

plt.plot(t, x)
plt.plot(mint, minx, ".")
plt.plot(maxt, maxx, ".")
plt.show()

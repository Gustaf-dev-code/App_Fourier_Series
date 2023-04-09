import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image

st.title("Fourier Serier WebApp")
st.markdown(
    "This application is made por the subject of Communications 2 of the National Pedagogical University." +
    " With which, the graphing of different functions is proposed applying the trigonometric and complex Fourier series.")
st.caption("Developed by Gustavo Briceño")

st.header("Trigonometric Fourier Series")
img = Image.open('fourier.jpg')
st.image(img, width=700)
st.latex(
    r'''f(t)= \frac{a_{0}}{2} + \sum_{n=1}^{\infty}[a_{n}cos(nw_{0}t) + b_{n}sin(nw_{0}t)]''')

st.subheader("Select the type of signal that you want to simulate.")
signal_type = st.selectbox(
    "", options=["Exponential", "Triangular", "Square", "Rectified Sinusoidal", "Trapezoidal", "Quadratic", "Sine", "Cosine"], label_visibility="hidden")
if(signal_type == "Exponential"):
    st.latex(r'''f(t) = Ae^{-kt}''')
if(signal_type == "Triangular"):
    st.latex(r'''f(t) = A*sawtooth(\frac{2π}{T}t,0.5)''')
if(signal_type == "Square"):
    st.latex(r'''f(t) = A*square(\frac{2π}{T}t,0.5)''')
if(signal_type == "Rectified Sinusoidal"):
    st.latex(r'''f(t) = A|Sin(w_{0}t)|''')
if(signal_type == "Trapezoidal"):
    st.latex(r'''f_{1}(t) = t,[0 < t < \frac{T}{3}]''')
    st.latex(r'''f_{2}(t) = \frac{T}{3},[\frac{T}{3} < t < \frac{2T}{3}]''')
    st.latex(r'''f_{3}(t) = -t,[\frac{2T}{3} < t < T]''')
if(signal_type == "Quadratic"):
    st.latex(r'''f(t) = t^{2}''')
if(signal_type == "Sine"):
    st.latex(r'''f(t) = A*Sin(w_{0}t)''')
if(signal_type == "Cosine"):
    st.latex(r'''f(t) = A*Cos(w_{0}t)''')

# In
T = st.number_input("Enter period [T]", 1, key="1")
n = st.number_input("Enter the number of harmonics [n]", 2, key="2")
A = st.number_input("Enter the amplitude [A]", 1)
n = int(n)
dt = 0.005
w0 = (2*np.pi)/T
signal_size = int((2*T+dt)/dt)

# Out
an = [0]*n
bn = [0]*n
an = np.zeros(n)
bn = np.zeros(n)
a0 = 0
pc = 0
maxi = A

if(signal_type == "Exponential"):
    t = np.arange(0, T+dt, dt)
    k = st.number_input("Enter decay factor", 1, key="3")
    f = A*(np.exp(-k*t))
if(signal_type == "Triangular"):
    t = np.arange(0, T+dt, dt)
    f = A*signal.sawtooth(w0*t, 0.5)
if(signal_type == "Square"):
    t = np.arange(0, 2*T+dt, dt)
    f = A*signal.square(w0*t, 0.5)
    pc = A*0.35
if(signal_type == "Rectified Sinusoidal"):
    t = np.arange(0, 2*T+dt, dt)
    f = A*abs(np.sin(w0*t))
if(signal_type == "Trapezoidal"):
    t = np.arange(0, T, dt)
    periodo = T/3

    def piece1(z):
        return z

    def piece2():
        return periodo

    def piece3(z):
        return -z

    f = np.piecewise(t, [(t >= 0) & (t < periodo),
                     (t >= periodo) & (t < 2*periodo), (t >= 2*periodo) & (t < 3*periodo)], [lambda t:t, lambda t:T/3, lambda t: -t+T])

    piece1 = np.vectorize(piece1)
    piece2 = np.vectorize(piece2)
    piece3 = np.vectorize(piece3)

    maxi = periodo

if(signal_type == "Quadratic"):
    t = np.arange(0, T+dt, dt)
    f = t*t
    maxi = T**2
if(signal_type == "Sine"):
    t = np.arange(0, T, dt)
    f = A*np.sin(w0*t)
if(signal_type == "Cosine"):
    t = np.arange(0, T, dt)
    f = A*np.cos(w0*t)


m = len(t)

btn = st.button("Perform Fourier series")
btn2 = st.button("See table of Fourier values")

maxtotal = 0
Cn = np.zeros(n)
phase = np.zeros(n)

# Para a0
for i in range(1, m):
    a0 += (1/T)*f[i]*dt

# Para an, bn, Cn y phase
for i in range(1, n, 1):
    for j in range(1, m, 1):
        an[i] += (2/T)*f[j]*np.cos(i*t[j]*w0)*dt
        bn[i] += (2/T)*f[j]*np.sin(i*t[j]*w0)*dt

    Cn[i] = (((an[i])**2)+((bn[i])**2))**(1/2)
    phase[i] = np.arctan((bn[i])/(an[i]))*(-1)
    maxtotal += Cn[i]


fourier_serie = a0
t1 = np.arange(0, 2*T+dt, dt)
tAf = np.arange(1, n+1, 1)
if(btn):
    fourier_serie += an[1]*np.cos(1*w0*t1)+bn[1]*np.sin(1*w0*t1)
    fig = plt.figure(figsize=(8, 8))
    ax2 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    ax1.set_title("Fourier Series")
    ax2.set_title("Original Signal")
    ax1.set_xlabel("Time [sec]")
    ax1.set_ylabel("Amplitude")
    ax2.set_xlabel("Time [sec]")
    ax2.set_ylabel("Amplitude")
    ax1.plot(t1, fourier_serie)
    ax2.plot(t, f)
    ax1.grid(color='gray', linestyle='dotted')
    ax2.grid(color='gray', linestyle='dotted')
    fig.tight_layout()
    plots1 = st.pyplot(fig)
    fig3 = plt.figure(figsize=(8, 8))
    ax3 = fig3.add_subplot(2, 1, 1)
    ax3.set_title("Amplitude Spectrum")
    ax3.set_xlabel("Harmonics")
    ax3.set_ylabel("Amplitude")
    ax3.stem(tAf, Cn)
    ax3.grid(color='gray', linestyle='dotted')
    ax4 = fig3.add_subplot(2, 1, 2)
    ax4.set_title("Phase Spectrum")
    ax4.set_xlabel("Harmonics")
    ax4.set_ylabel("...")
    ax4.stem(tAf, phase)
    ax4.grid(color='gray', linestyle='dotted')
    fig3.tight_layout()
    st.pyplot(fig3)
    for i in range(2, n, 1):
        fourier_serie += an[i]*np.cos(i*w0*t1)+bn[i]*np.sin(i*w0*t1)
        max2 = max(fourier_serie)
        max2 -= pc
        max3 = max2/maxi
        max_final = fourier_serie/max3

        fig = plt.figure(figsize=(8, 8))
        ax2 = fig.add_subplot(2, 1, 1)
        ax1 = fig.add_subplot(2, 1, 2)
        ax1.set_title("Fourier Series")
        ax2.set_title("Original Signal")
        ax1.set_xlabel("Time [sec]")
        ax1.set_ylabel("Amplitude")
        ax2.set_xlabel("Time [sec]")
        ax2.set_ylabel("Amplitude")
        ax1.plot(t1, max_final)
        ax2.plot(t, f)
        ax1.grid(color='gray', linestyle='dotted')
        ax2.grid(color='gray', linestyle='dotted')
        ax1.axvline(0, color='gray')
        ax1.axhline(0, color='gray')
        ax2.axvline(0, color='gray')
        ax2.axhline(0, color='gray')
        plots1.pyplot(fig)

if(btn2):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    Cn_total = 0
    for i in range(1, n, 1):
        for j in range(1, m, 1):
            an[i] += (2/T)*f[j]*np.cos(i*t[j]*w0)*dt
            bn[i] += (2/T)*f[j]*np.sin(i*t[j]*w0)*dt
            list1.append(an[i])
            list2.append(bn[i])
        Cn[i] = (((an[i])**2)+((bn[i])**2))**(1/2)
        phase[i] = np.arctan((bn[i])/(an[i]))*(-1)
        list3.append(Cn[i])
        list4.append(phase[i])
    max_length = max(len(list1), len(list2), len(list3), len(list4))
    while len(list1) < max_length:
        list1.append(None)
    while len(list2) < max_length:
        list2.append(None)
    while len(list3) < max_length:
        list3.append(None)
    while len(list4) < max_length:
        list4.append(None)

    data = {"an": list1, "bn": list2, "Cn": list3, "Phase": list4}
    df = pd.DataFrame(data)
    st.table(df)

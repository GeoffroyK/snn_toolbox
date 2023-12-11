from pylab import *
import matplotlib.pyplot as plt
rcParams['figure.figsize']=(12,3) # Change the default figure size

I=1                         #Set the parameter I.
C=1                          #Set the parameter C.
Vth = 1;                     #Define the voltage threshold.
Vreset = 0;                  #Define the reset voltage.
dt=0.01                      #Set the timestep.
V = zeros([1000,1])          #Initialize V.
V[0]=0.2;                    #Set the initial condition.

for k in range(1,999):       #March forward in time,
    V[k+1] = V[k] + dt*(I/C) #Update the voltage,
    if V[k+1] > Vth:         #... and check if the voltage exceeds the threshold.
        V[k+1] = Vreset
        
t = arange(0,len(V))*dt      #Define the time axis.

figure()                     #Plot the results.
plot(t,V)
xlabel('Time [s]')
ylabel('Voltage [mV]')
show()


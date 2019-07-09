import numpy as np
import nengo
import matplotlib.pyplot as plt
from scipy.integrate import quad
from nengo.utils.ensemble import tuning_curves
import pdb
from nengo.utils.functions import piecewise

#Lamprey specs
l = 1.0    # Length of lamprey in z direction
l3 = 1.0   # Length of lamprey segment in y direction
w = 30     #forward angular velocity omega
n = 3      #Number of segments
I_p = 1    #Moment of inertia about pitch axis.

#Model params
alpha_damp = -0.3    #damping parameter for higher order terms in M_d
T = 5                #Simulation time in seconds
sigma = 0.1         #Standard deviation of gaussian encoding function
num_neurons = 1000   #number of neurons per dimension
tau = 0.01           #Synaptic time constant

#variables needed for roll motion
time_start = 0             #the time at which the rolling input was triggered
roll_triggered = False     #boolean indicating whether rolling motion has been triggered

#Find the z coordinate of the segment centers
segment_centers = np.zeros([n,1])
for i in range(n):
    segment_centers[i] = l*(1+2*i)/(2.0*n)
        
#Find M_dynamics matrix
M_A = np.array([[0,w,0],[-w,0,0],[0,-w,0]])
M_damp = np.array([[alpha_damp, 0, alpha_damp],[0, 0, 0], [alpha_damp, 0, alpha_damp]])
M_d = M_A + M_damp

#Find M_I matrix
M_I = np.array([[8,0,-8],[0,1,0],[-8,0,8]])

#Calculate theta to convert to activity space
def gauss_encoder(z,segment_center_z,sigma=0.01):
    return 1/(sigma*(2*np.pi)**0.5) * np.exp(-1*np.square(z-segment_center_z)/(2*sigma**2))

def phi(z):
    return [1, np.sin(2*np.pi*z),np.cos(2*np.pi*z)]

encoders = np.zeros([n,1])
phi_mat = np.zeros([1,3])

theta_inv = np.zeros([n,3])

for i in range(n):
    for j in range(3):
        integrate_func = lambda z: phi(z)[j] * gauss_encoder(z,segment_centers[i],sigma)
        theta_inv[i,j] = quad(integrate_func,0,l)[0]

theta = np.linalg.pinv(theta_inv)

#calculate md,mi matrix
m_d = np.matmul(np.matmul(theta_inv,M_d),theta)
m_i = np.matmul(np.matmul(theta_inv,M_I),theta)

#Create the nengo model
model = nengo.Network('Simple model', seed=3)

A_prime = tau*m_d + np.eye(n)
B_prime = tau*m_i

with model:
        
    #Function that is used in the recurrent connection of forward dynamics. 
    #Dynamically calculates the dynamics matrix and transforms the input, x
    def calc_md(x):
        w = x[0]*100
        activity = np.array(x[1:])
        
        M_A = np.array([[0,w,0],[-w,0,0],[0,-w,0]])
        M_damp = np.array([[alpha_damp, 0, alpha_damp],[0, 0, 0], [alpha_damp, 0, alpha_damp]])
        M_d = M_A + M_damp
        m_d = np.matmul(np.matmul(theta_inv,M_d),theta)
        A_prime = tau*m_d + np.eye(n)
        
        return np.matmul(A_prime,activity)
    
    #Given an input, x = [a,b,c], returns [0,a,b,c] 
    def transform_a(x):
        a = np.zeros([n+1,n])
        ind = 0
        for i in range(n+1):
            if i == 0:
                continue
            a[i,ind] = 1
            ind = ind + 1
    
        return np.matmul(a,x)
    
    #Given an input, x = [w], returns [w,0,0,0]
    def transform_omega(x):
        a = np.zeros([n+1,1])
        a[0] = 1
        #pdb.set_trace()
        return np.matmul(a,x)
    
    #Given the current time step and omega, it calculates the theoretical tension
    def calc_tension(t,om_in):
        omega = om_in * 100.0
        T = []
        for i in range(n):
            T = np.append(T,np.sin(omega*t - 2*np.pi*segment_centers[i]) - np.sin(omega*t),axis=0)
        return T
    
    #Given, muscle activity, it calculates muscle tension by multiplying and summing up with gaussian
    def calc_decoded_tension(x):
        T = [] 
        for i in range(n):
            t = 0
            for j in range(n):
                t += gauss_encoder(segment_centers[i],segment_centers[j],sigma) * x[j] 
            T = np.append(T,t,axis=0)
        
        return T
    
    #Provides the model with the startup stimulus (current A) until t = 0.2s
    def calc_startup_stim(t,a):
        if t < 0.2:
            return np.matmul(B_prime,a)
        else:
            return np.zeros(n)
    
    #Given a vector of activities for segments, returns the activity for the first segment
    def calc_muscle_a(a):
        return a[0]
    
    def yaw_input(t):
        gradient = 0.0007
        f = piecewise({0:0.0,2.5:gradient*t - gradient*2.5,3.5:gradient*3.5 - gradient*2.5})
        return f(t)
        
    def omega_input(t):
        f = piecewise({0:0.2, 2.5:0.3*t-0.55, 3: 0.35})
        w = f(t)*100
        return f(t)
        
    #Converts from orthonormal basis space to activity space
    def calc_yaw(x):
        C_prime = np.matmul(theta_inv,[1,w,0])
        return C_prime*x
    
    #Transforms the input pitch acceleration, x to a force
    def inertia_calc(x):
        return x*np.ones([n])*I_p/l3
    
    #Transforms muscle tensions to muscle activity by multiplying with gaussian function
    def calc_muscle_activity(x):
        A = [] 
        for i in range(n):
            t = 0
            for j in range(n):
                t += gauss_encoder(segment_centers[i],segment_centers[j],sigma) * x[j] 
            A = np.append(A,t,axis=0)

        return A
    
    #Yaw+forward motion nodes
    
    #Yaw acceleration inputs
    yaw_acceleration = nengo.Node(yaw_input, size_out = 1)
    
    #Startup stimulus for the oscillator
    startup_stim = nengo.Node(calc_startup_stim,size_in = n)
    
    #A node to calculate theoretical tension
    theoretical_tension = nengo.Node(calc_tension,size_in=1)
    
    #Forward angular velocity inputs
    omega_in = nengo.Node(0.2,size_out=1)
    
    #Yaw + forward ensembles
    
    #Ensemble to represent muscle activities
    ens_a = nengo.Ensemble(num_neurons, dimensions=n,radius=3)
    
    #Ensemble to represent muscle tensions
    ens_tension = nengo.Ensemble(num_neurons,dimensions=n)
    
    #Ensemble to dynamically calculate the dynamics matrix
    ens_sum = nengo.Ensemble(num_neurons*(n+1),dimensions=n+1)
    
    #Ensemble to represent muscle activities in the right and left sides of the lamprey
    ens_muscle_a_right = nengo.Ensemble(num_neurons,encoders = np.ones([num_neurons,1]), dimensions=1)
    ens_muscle_a_left = nengo.Ensemble(num_neurons,encoders = -1*np.ones([num_neurons,1]), dimensions=1)
    
    #Yaw + forward motion connections and probes
    
    nengo.Connection(omega_in,theoretical_tension)
    nengo.Connection(yaw_acceleration,ens_a,function=calc_yaw,synapse=tau)
    nengo.Connection(omega_in,ens_sum,function=transform_omega,synapse=None)
    nengo.Connection(ens_a,startup_stim)
    nengo.Connection(startup_stim,ens_a,synapse=tau)
    nengo.Connection(ens_a,ens_tension,function=calc_decoded_tension,synapse=tau)
    nengo.Connection(ens_a,ens_muscle_a_right,function = calc_muscle_a)
    nengo.Connection(ens_a,ens_muscle_a_left,function = calc_muscle_a)
    nengo.Connection(ens_a, ens_sum,function=transform_a,synapse=tau/2)
    nengo.Connection(ens_sum,ens_a,function=calc_md,synapse=tau/2)
    
    omega_probe = nengo.Probe(omega_in)
    yaw_probe = nengo.Probe(yaw_acceleration)
    muscle_a_probe_left = nengo.Probe(ens_muscle_a_left.neurons,'spikes',synapse=tau)
    muscle_a_probe_right = nengo.Probe(ens_muscle_a_right.neurons,'spikes',synapse=tau)
    
    decoded_t_probe = nengo.Probe(ens_tension, synapse=tau)
    theoretical_t_probe = nengo.Probe(theoretical_tension, synapse=tau)
    a = nengo.Probe(ens_a, synapse=tau)

    #Pitch motion nodes and ensembles
    
    #A node representing a constant of 0
    const_0 = nengo.Node(np.zeros([n]))
    
    #A node for the pitch input
    pitch_in = nengo.Node(piecewise({0:0.4, 0.5:1, 1:0}))
    
    #An ensemble to represent x1
    ens_p = nengo.Ensemble(num_neurons, dimensions=n)
    
    #An ensemble to represent pitching muscle tensions
    ens_pitch_a = nengo.Ensemble(num_neurons,dimensions=n)
    
    #An ensemble to represent pitching muscle activity
    ens_pitch_activity = nengo.Ensemble(num_neurons, dimensions = n,radius = 5)
    
    #pitching motion conncections and probes
    nengo.Connection(const_0,ens_p)
    nengo.Connection(ens_p,ens_p,synapse=tau)
    nengo.Connection(pitch_in,ens_pitch_a,function = inertia_calc,synapse=tau)
    nengo.Connection(ens_p,ens_pitch_a,synapse=tau)
    nengo.Connection(ens_pitch_a,ens_pitch_activity,function=calc_muscle_activity,synapse=tau)

    p = nengo.Probe(ens_pitch_a, synapse=tau)
    pitch_in_probe = nengo.Probe(pitch_in,synapse = tau)
    p_activity_probe = nengo.Probe(ens_pitch_activity,synapse=tau)
    
    #Given t and roll angular velocity, it calculates u(t), heaviside function
    def roll_heaviside_calc(t, roll_acc):
        global roll_triggered,time_start
        
        if abs(roll_acc - 0) < 1e-10:
            #reset trigger if input is 0
            roll_triggered = False
            return np.zeros([n])
        elif roll_triggered == False and abs(roll_acc) > 0:
            #set trigger as soon as theres an input
            time_start = t
            roll_triggered = True
            return np.zeros([n])
        elif roll_triggered == True:
            a = np.zeros([n])
            for i in range(n):
                a[i] = roll_acc * np.heaviside(t - time_start - segment_centers[i] / l + segment_centers[0],0)  
            return a
    
    #Roll motion ensembles and nodes
    
    #Node to input roll angular velocity
    roll_in = nengo.Node(piecewise({0:0, 0.5:0.5, 2:0}))
    
    #Node that inputs a constant 0 
    const_0_roll = nengo.Node(np.zeros([n]))
        
    #Node to calculate heaviside function
    roll_heaviside = nengo.Node(roll_heaviside_calc,size_in = 1, size_out = n)
    
    #Ensemble that represents x2
    ens_r = nengo.Ensemble(num_neurons, dimensions=n)
    
    #Ensemble that represents rolling muscle tensions
    ens_roll_t = nengo.Ensemble(num_neurons,dimensions=n)
    
    #Ensemble that represents rolling muscle activity
    roll_muscle_act = nengo.Ensemble(num_neurons, dimensions = n,radius=5)
    
    #Rolling motion connections and probes
    nengo.Connection(ens_roll_t,roll_muscle_act,function=calc_muscle_activity)
    nengo.Connection(roll_in, roll_heaviside)
    nengo.Connection(roll_heaviside, ens_roll_t)
    nengo.Connection(const_0_roll,ens_r)
    nengo.Connection(ens_r,ens_roll_t)
    nengo.Connection(ens_r,ens_r,synapse=tau)
    
    roll_activity_probe = nengo.Probe(roll_muscle_act,synapse = tau)
    roll_in_probe = nengo.Probe(roll_in,synapse = tau)
        

sim = nengo.Simulator(model)
sim.run(T)


def plot(x,y,title,xlab,ylab):
    plt.figure()
    plt.plot(x,y)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

def activity_plot(x,y,title,xlab,ylab):
    plt.figure()
    for i in range(n):
        plt.plot(x,y[:,i],label='z=' + str(segment_centers[i]))
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    
#Yaw + forward motion plots
plot(sim.trange(),sim.data[omega_probe],'Omega input','Time (s)','Omega/100')
plot(sim.trange(),sim.data[yaw_probe],'Yaw input','Time (s)','Yaw acceleration')
activity_plot(sim.trange(), sim.data[decoded_t_probe],'Plot of decoded tensions','time (s)','T')
activity_plot(sim.trange(), sim.data[a],'Segment activities','Time (s)','Activity')

plt.figure()
avg1 = np.average(sim.data[muscle_a_probe_left],axis=1)
avg2 = np.average(sim.data[muscle_a_probe_right],axis=1)
plt.plot(sim.trange(),avg1,label='left neurons')
plt.plot(sim.trange(),avg2,label='right neurons')
plt.title('Average firing rate (Hz) for first segment')
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (Hz)')
plt.legend()

#Pitch plots
plot(sim.trange(),sim.data[pitch_in_probe],'Pitch input','Time (s)','Pitch angular acceleration')
activity_plot(sim.trange(),sim.data[p_activity_probe],'Pitch muscle activity','Time (s)','Activity')

#Roll plots
plot(sim.trange(),sim.data[roll_in_probe],'Roll acceleration input','Time (s)','Acceleration')
activity_plot(sim.trange(), sim.data[roll_activity_probe],'Roll muscle activity','Time (s)','Activity')
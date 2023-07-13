import SDE_tools,SDE_graphs

#________________
#Input Parameters
#________________
mode = 'gamma'          #Equation on ω : 'gamma' or 'lognormal'
N = 500                 #Number of particles
Nt = 500                #Number of timesteps
dt = 0.05               #Timestep duration
uv_mean = 0             #Initial mean value of uv
uv_std = 100            #Standard Deviation of uv
omega_mean = 0.100      #Initial mean value of ω
sph_means = True        #Mean computation mode : True = SPH means / False = global means
save_simulation = True  #Save simulation result
plot = True             #Plot figures
save_plots = True       #Save plots
load_simulation = False #Simulation mode : True = load a previous saved simulation / False = run a new simulation

#__________
#Simulation
#__________
if load_simulation:
    result,means,N,Nt,dt = SDE_tools.load_result(mode,N,Nt,omega_mean) #Load a previous saved simulation
else:
    result,dW = SDE_tools.init(mode,N,Nt,dt,uv_mean,uv_std,omega_mean) #Initiate fluide quantities
    result,means = SDE_tools.run(mode,result,N,Nt,dt,omega_mean,dW,save=save_simulation,output_step=25,sph_means=sph_means) #Integrate SDEs

#_____
#Plots
#_____
if plot:
    SDE_graphs.paths_graph(          result      ,(Nt,N,dt,mode),save=save_plots) #Plot random walks of ω
    SDE_graphs.means_graph(          result      ,(Nt,N,dt,mode),save=save_plots) #Plot global mean values of x,uv,uh,ω
    SDE_graphs.SPH_means_graph(      result,means,(Nt,N,dt,mode),save=save_plots) #Plot SPH mean values of x,uv,uh,ω
    SDE_graphs.pdfs_graph(           result      ,(Nt,N,dt,mode),save=save_plots) #Plot PDFs of uv,ω
    SDE_graphs.autocorrelation_graph(result      ,(Nt,N,dt,mode),save=save_plots) #Plots autocorrelation functions of uv and ω
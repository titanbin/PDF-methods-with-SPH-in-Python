import numpy as np
from tqdm import tqdm,trange

C_0 = 2.1
C_chi = 1.6
C_3 = 1
Delta_C_0 = 3.5-2.1
L = 1

def predictor(funcs,y,dt,dW,means):
    #______________
    #Predictor Step
    #______________
    f,g,mod = funcs
    N = len(y)
    res = np.zeros((N,4))
    for i in range(N):
        pred_y = y[i] + f(y[i],means[:,i]) * dt + g(y[i],means[:,i]) * dW[i]

        if pred_y[3] < 0:
            pred_y[3] = y[i,3]

        res[i] = pred_y
    return res

def corrector(funcs,y,pred_y,dt,dW,means,pred_means,alpha=0.5,beta=0.5):
    #______________
    #Corrector step
    #______________
    f,g,mod = funcs
    N = len(y)
    res = np.zeros((N,4))
    for i in range(N):      
        pred_y_a = f(y[i],means[:,i]) - 0.5*mod(y[i],means[:,i])
        corr_y_a = f(pred_y[i],pred_means[:,i]) - 0.5*mod(pred_y[i],pred_means[:,i])

        new_y = (y[i] + ((1-alpha)*pred_y_a + alpha*corr_y_a)*dt + ((1-beta)*g(y[i],means[:,i]) + beta*g(pred_y[i],pred_means[:,i]))*dW[i])

        new_y[0] = new_y[0] % L#Periodic conditions

        if new_y[3] < 0:
            new_y[3] = y[i,3]

        res[i] = new_y
    return res

def f_lognormal(y,means):
    #_________________
    #Drift Coefficient
    #_________________
    x,uv,uh,omega = y
    rho_mean,uv_mean,uh_mean,omega_mean,k_mean,sigma_mean,So_mean = means

    drift_coefficient = np.array((
        uv,
        -(0.5*omega_mean + 0.75*(C_0*omega + Delta_C_0*omega_mean))*(uv-uv_mean),
        -(0.5*omega_mean + 0.75*(C_0*omega + Delta_C_0*omega_mean))*uh,
        -omega*omega_mean*C_chi*(np.log(omega/omega_mean)-sigma_mean)
    ))
    return drift_coefficient

def g_lognormal(y,means):
    #_____________________
    #Diffusion Coefficient
    #_____________________
    x,uv,uh,omega = y
    rho_mean,uv_mean,uh_mean,omega_mean,k_mean,sigma_mean,So_mean = means

    if sigma_mean < 0:
        sigma_mean = 0.5

    diffusion_coefficient = np.array((
        0,
        np.sqrt(C_0*k_mean*omega),
        np.sqrt(C_0*k_mean*omega),
        omega*np.sqrt(2*C_chi*omega_mean*2*sigma_mean)
    ))
    return diffusion_coefficient

def mod_lognormal(y,means):
    #__________________________
    #Modified Drift Coefficient
    #__________________________
    x,uv,uh,omega = y
    rho_mean,uv_mean,uh_mean,omega_mean,k_mean,sigma_mean,So_mean = means

    modified_drift_coefficient = np.array((
        0,
        0,
        0,
        omega*2*C_chi*omega_mean*2*sigma_mean
    ))
    return modified_drift_coefficient

def f_gamma(y,means):
    #________________
    #Drift Coeficient
    #________________
    x,uv,uh,omega = y
    rho_mean,uv_mean,uh_mean,omega_mean,k_mean,omega_mean_conditional,So_mean = means

    drift_coefficient = np.array((
        uv,
        -(0.5 + 0.75*C_0)*omega*(uv-uv_mean),
        -(0.5 + 0.75*C_0)*omega*uh,
        -omega_mean_conditional*(omega-omega_mean)*C_3
    ))
    return drift_coefficient

def g_gamma(y,means):
    #_____________________
    #Diffusion Coefficient
    #_____________________
    x,uv,uh,omega = y
    rho_mean,uv_mean,uh_mean,omega_mean,k_mean,omega_mean_conditional,So_mean = means

    diffusion_coefficient = np.array((
        0,
        np.sqrt(C_0*k_mean*omega),
        np.sqrt(C_0*k_mean*omega),
        np.sqrt(2*1/4*C_3*omega_mean*omega_mean_conditional*omega)
    ))
    return diffusion_coefficient

def mod_gamma(y,means):
    #__________________________
    #Modified Drift Coefficient
    #__________________________
    x,uv,uh,omega = y
    rho_mean,uv_mean,uh_mean,omega_mean,k_mean,omega_mean_conditional,So_mean = means

    modified_drift_coefficient = np.array((
        0,
        0,
        0,
        0.5*2*1/4*C_3*omega_mean*omega_mean_conditional
    ))
    return modified_drift_coefficient

def lognormal_distribution(mean,N):
    #_____________________________________________________________
    #Generate lognormal ω distribution with given mean and STD = 1
    #_____________________________________________________________
    nu = np.log(mean**2/np.sqrt(2*(mean**2)))
    sigma = np.sqrt(np.log(2))
    return np.exp(np.random.normal(nu,sigma,N))

def gamma_distribution(mean,N):
    #______________________________________________________________
    #Generate gamma ω distribution with given mean and σ² = 0.25<ω>
    #______________________________________________________________
    theta = 0.25
    k = 4
    d = np.random.default_rng().gamma(k,theta,N)
    d *= mean
    return d

def compute_means(mode,y,delta_M):
    #__________________________________________
    #Compute global mean quantity terms in SDEs
    #__________________________________________
    N = y.shape[0]
    rho_mean = np.ones(N)*np.mean(y[:,0])
    uv_mean = np.ones(N)*np.mean(y[:,1])
    uh_mean = np.ones(N)*np.mean(y[:,2])
    omega_mean = np.ones(N)*np.mean(y[:,3])
    if mode == 'lognormal':
        sigma_mean = np.mean(y[:,3]/omega_mean*np.log(y[:,3]/omega_mean))
    elif mode == 'gamma':
        sigma_mean = np.ma.array(y[:,3],mask=y[:,3] < omega_mean).mean()
    sigma_mean = np.ones(N)*sigma_mean
    k_mean = 0.5*np.mean((y[:,1]-uv_mean)**2) + np.mean((y[:,2]-uh_mean)**2)
    k_mean = np.ones(N)*k_mean
    return np.array((rho_mean,uv_mean,uh_mean,omega_mean,k_mean,sigma_mean))

def SPH_means(mode,y,delta_M):
    #_______________________________________
    #Compute SPH mean quantity terms in SDEs
    #_______________________________________
    N = y.shape[0]
    h = L*N**(-1/4)
    x = y[:,0]
    uv = y[:,1]
    uh = y[:,2]
    omega = y[:,3]

    rho_mean = np.zeros(N)
    uv_mean = np.zeros(N)
    uh_mean = np.zeros(N)
    omega_mean = np.zeros(N)
    k_mean = np.zeros(N)
    sigma_mean = np.zeros(N)
    duv_mean = np.zeros(N)
    duh_mean = np.zeros(N)
    uvuh_mean = np.zeros(N)
    uvuv_mean = np.zeros(N)

    for i in range(N):
        for j in range(N):
            r = abs(x[i]-x[j])
            if r > L/2:
                r = abs(r-L)
            K = Kernel(r,h)
            rho_mean[i] += K
            uv_mean[i] += uv[j]*K
            uh_mean[i] += uh[j]*K
            omega_mean[i] += omega[j]*K
            dK = dKernel(r,h)
            duv_mean[i] += uv[j]*dK
            duh_mean[i] += uh[j]*dK
        
        rho_mean[i] *= delta_M
        uv_mean[i] *= delta_M/rho_mean[i]
        uh_mean[i] *= delta_M/rho_mean[i]
        omega_mean[i] *= delta_M/rho_mean[i]
        duv_mean[i] *= delta_M/rho_mean[i]
        duh_mean[i] *= delta_M/rho_mean[i]

    k = 0.5*(uv-uv_mean)**2 + (uh-uh_mean)**2
    uvuh = (uv-uv_mean)*(uh-uh_mean)
    uvuv = (uv-uv_mean)**2
    if mode == 'lognormal':
        sigma = omega/omega_mean*np.log(omega/omega_mean)
    elif mode == 'gamma':
        sigma = omega_mean

    for i in range(N):
        for j in range(N):
            r = abs(x[i]-x[j])
            if r > L/2:
                r = abs(r-L)
            K = Kernel(r,h)
            k_mean[i] += k[j]*K
            sigma_mean[i] += sigma[j]*K
            uvuh_mean[i] += uvuh[j]*K
            uvuv_mean[i] += uvuv[j]*K
        
        k_mean[i] *= delta_M/rho_mean[i]
        sigma_mean[i] *= delta_M/rho_mean[i]
        uvuh_mean[i] *= delta_M/rho_mean[i]
        uvuv_mean[i] *= delta_M/rho_mean[i]

    Pii_mean = -2*uvuv_mean*duv_mean - 4*uvuh_mean*duh_mean
    So_mean = (1.90-1) + (1.45 - 1)*Pii_mean/(omega_mean*k_mean)

    return np.array((rho_mean,uv_mean,uh_mean,omega_mean,k_mean,sigma_mean,So_mean))

def Kernel(r,h):
    #__________________________
    #Polynomial Kernel Function
    #__________________________
    if r > h:
        return 0
    return 5/(4*h)*(1 + 3*r/h)*(1 - r/h)**3

def dKernel(r,h):
    #________________________________________
    #Derivative of polynomial Kernel Function
    #________________________________________
    if r > h:
        return 0
    return -5/(4*h)*12.0*(r/h - 2.0*(r/h)**2 + (r/h)**3)/h

def init(mode,N,Nt,dt,uv_mean,uv_std,omega_mean):
    #_________________________________________
    #Initiate quantities and declare variables
    #_________________________________________
    global C_0
    result = np.zeros((Nt,N,4))
    dW = np.random.normal(0,np.sqrt(dt),(Nt,N,4))

    result[0,:,0] = np.random.uniform(0,1,N)

    result[0,:,0] = np.linspace(0,1,N+1)[:-1]
    
    result[0,:,1] = np.random.normal(uv_mean,uv_std,N)
    result[0,:,2] = np.random.normal(0,1,N)

    if mode == 'lognormal':
        result[0,:,3] = lognormal_distribution(omega_mean,N)
    elif mode == 'gamma':
        result[0,:,3] = gamma_distribution(omega_mean,N)
    
    return result,dW

def run(mode,result,N,Nt,dt,omega_mean,dW,save=False,output_step=0,sph_means=True):
    #________________
    #SDEs integration
    #________________
    result_means = np.zeros((Nt,7,N))
    if sph_means:
        result_means[0] = SPH_means(mode,result[0],1/N)
    else:
        result_means[0] = compute_means(mode,result[0],1/N)

    if mode == 'lognormal':
        funcs = (f_lognormal,g_lognormal,mod_lognormal)
    elif mode == 'gamma':
        funcs = (f_gamma,g_gamma,mod_gamma)
    for t in trange(1,Nt):
        #Compute mean quantities for Predictor step
        if sph_means:
            means = SPH_means(mode,result[t-1],1/N)
        else:
            means = compute_means(mode,result[t-1],1/N)

        #Predictor Step
        pred_result = predictor(funcs,result[t-1],dt,dW[t],means)
        #Compute mean quantities for Corrector step
        if sph_means:
            means2 = SPH_means(mode,pred_result,1/N)
        else:
            means2 = compute_means(mode,pred_result,1/N)
        #Corrector Step
        result[t] = corrector(funcs,result[t-1],pred_result,dt,dW[t],means,means2)
        result_means[t] = means2

        if output_step != 0 and t % output_step == 0:
            tqdm.write("{}/{} | <uv(t)> - <uv(0)> = {:.4f}, <uv(t)> = {:.4f}".format(t,Nt,np.mean(result[t,:,1])-np.mean(result[0,:,1]),np.mean(result[t,:,1])))
            tqdm.write("{}/{} | <w(t)> - <w(0)> = {:.4f}, <w(t)> = {:.4f}".format(t,Nt,np.mean(result[t,:,3])-np.mean(result[0,:,3]),np.mean(result[t,:,3])))

    if save:
        np.savez('results/{}_N{}k_Nt{}k_w{:.4f}.npz'.format(mode,int(N/1000),int(Nt/1000),omega_mean),[N,Nt,dt],result,result_means)

    return result,result_means

def load_result(mode,N,Nt,omega_mean):
    #_____________________
    #Load saved simulation
    #_____________________
    file = np.load('results/{}_N{}k_Nt{}k_w{:.4f}.npz'.format(mode,int(N/1000),int(Nt/1000),omega_mean))
    N,Nt,dt = file['arr_0']
    result = file['arr_1']
    means = file['arr_2']
    return result,means,int(N),int(Nt),dt
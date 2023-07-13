import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as grid_spec
from matplotlib.ticker import ScalarFormatter

def means_graph(y,parameters,save=False):
    #____________________________________
    #Plot global mean values of x,uv,uh,ω
    #____________________________________
    Nt,N,dt,mode = parameters
    fig,ax = plt.subplots(2,2,figsize=(10,10))
    t = np.linspace(0,Nt*dt,Nt)
    ax[0,0].plot(t,np.mean(y[:,:,0],axis=1))
    ax[0,1].plot(t,np.mean(y[:,:,1],axis=1))
    ax[1,0].plot(t,np.mean(y[:,:,2],axis=1))
    ax[1,1].plot(t,np.mean(y[:,:,3],axis=1))
    ax[0,0].set_ylim(0,1)
    ax[0,1].ticklabel_format(useOffset=False)
    ax[1,0].ticklabel_format(useOffset=False)
    ax[1,1].ticklabel_format(useOffset=False)
    ax[0,0].set_xlabel('t')
    ax[0,1].set_xlabel('t')
    ax[1,0].set_xlabel('t')
    ax[1,1].set_xlabel('t')
    ax[0,0].set_ylabel('<x>')
    ax[0,1].set_ylabel(r'$<u_v>$')
    ax[1,0].set_ylabel(r'$<u_h>$')
    ax[1,1].set_ylabel(r'$<\omega>$')
    ax[0,0].grid()
    ax[0,1].grid()
    ax[1,0].grid()
    ax[1,1].grid()
    plt.tight_layout()
    if save:
        plt.savefig('figs/means_{}_N{}k_Nt{}k.png'.format(mode,int(N/1000),int(Nt/1000)),dpi=200)
    plt.show()

def paths_graph(y,parameters,save=False):
    #______________________
    #Plot random walks of ω
    #______________________
    Nt,N,dt,mode = parameters
    fig,ax = plt.subplots(5,1,figsize=(7,8))
    for i in range(5):
        omega = y[:,np.random.randint(0,N),3]
        t = np.linspace(0,Nt*dt/np.mean(omega),Nt)
        ax[i].plot(t,omega/np.mean(omega))
        ax[i].grid()
        if i == 4:
            ax[i].set_xlabel(r'$t/\tau$')
        else:
            ax[i].set_xticklabels([])
        ax[i].set_ylabel(r'$\frac{\omega}{\langle\omega\rangle}$')
        ax[i].set_xlim(0,5/np.mean(omega))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    if save:
        plt.savefig('figs/paths_{}_N{}k_Nt{}k.png'.format(mode,int(N/1000),int(Nt/1000)),dpi=200)
    plt.show()

def autocorrelation_graph(y,parameters,save=False):
    #___________________________________________
    #Plots autocorrelation functions of uv and ω
    #___________________________________________
    Nt,N,dt,mode = parameters
    uv = y[:,:,1]
    omega = y[:,:,3]
    uv_mean = np.mean(uv)
    omega_mean = np.mean(omega)
    fig,ax = plt.subplots(1,2,figsize=(15,7))
    def autocorr(x):
        result = np.correlate(x,x,mode='full')
        return result[result.size // 2:]

    corr_uv = np.zeros((N,Nt))
    corr_omega = np.zeros((N,Nt))
    for i in range(0,N):
        corr_uv[i,:] = autocorr(y[:,i,1] - uv_mean)
        corr_omega[i,:]=autocorr(y[:,i,3] - omega_mean)
    t = np.linspace(0,Nt*dt,Nt)
    
    if mode == 'lognormal':
        T_uv = 1/(1/2 + 3/4*3.5*np.mean(omega))
        T_chi = 1/(1.6*np.mean(omega))
        T_omega = T_chi*(1 - 2/9*np.var(omega))
    elif mode == 'gamma':
        T_uv = 1/(1/2 + 3/4*2.1*np.mean(omega))
        T_omega = 1/np.ma.array(omega,mask=omega < np.mean(omega)).mean()
    
    x = t[::Nt//250]
    y = np.mean(corr_uv,axis=0)/float(np.mean(corr_uv,axis=0).max())
    y = y[::Nt//250]
    ax[0].plot(t,np.exp(-t/T_uv),lw=5,label='uv théorique')
    ax[0].plot(x,y,'^',ms=8,label='uv simulation')
    
    x = t[::Nt//250]
    y = np.mean(corr_omega,axis=0)/float(np.mean(corr_omega,axis=0).max())
    y = y[::Nt//250]
    ax[1].plot(t,np.exp(-t/T_omega),lw=5,label='omega théorique')
    ax[1].plot(x,y,'^',ms=8,label='omega simulation')

    ax[0].legend()
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlabel('t')
    ax[1].set_xlabel('t')
    ax[0].set_ylabel(r'$\rho_{u_v}$')
    ax[1].set_ylabel(r'$\rho_{\omega}$')
    plt.tight_layout()
    if save:
        plt.savefig('figs/autocorrelation_{}_N{}k_Nt{}k.png'.format(mode,int(N/1000),int(Nt/1000)),dpi=200)
    plt.show()

def SPH_means_graph(y,means,parameters,save=False):
    #_________________________________
    #Plot SPH mean values of x,uv,uh,ω
    #_________________________________
    Nt,N,dt,mode = parameters
    fig,ax = plt.subplots(2,2,figsize=(10,10))

    x = y[:,:,0]

    rho_mean = means[:,0,:]
    uv_mean = means[:,1,:]
    uh_mean = means[:,2,:]
    omega_mean = means[:,3,:]

    sort_inds = x.argsort(axis=1)
    sorted_xs = np.take_along_axis(x,sort_inds,axis=-1)
    sorted_rho_mean = np.take_along_axis(rho_mean,sort_inds,axis=-1)
    sorted_uv_mean = np.take_along_axis(uv_mean,sort_inds,axis=-1)
    sorted_uh_mean = np.take_along_axis(uh_mean,sort_inds,axis=-1)
    sorted_omega_mean = np.take_along_axis(omega_mean,sort_inds,axis=-1)

    for t in range(25):
        alpha = 0.1
        if t == 0:
            alpha = 1
        i = len(sorted_xs)//25*t
        ax[0,0].plot(sorted_xs[i,:],sorted_rho_mean[i,:],'-',lw=3,alpha=alpha)
        ax[0,1].plot(sorted_xs[i,:],sorted_uv_mean[i,:],'-',lw=3,alpha=alpha)
        ax[1,0].plot(sorted_xs[i,:],sorted_uh_mean[i,:],'-',lw=3,alpha=alpha)
        ax[1,1].plot(sorted_xs[i,:],sorted_omega_mean[i,:],'-',lw=3,alpha=alpha)

    L = 1
    h = L*N**(-1/4)
    ymid = 0.75
    col = 'tab:red'
    p1 = patches.FancyArrowPatch((L/2-h/2,ymid),(L/2+h/2,ymid),arrowstyle='<->',mutation_scale=20,color=col,lw=3,zorder=10)
    ax[0,0].add_patch(p1)
    ax[0,0].vlines([L/2-h/2,L/2+h/2],0,2,ls=['--','--'],color=(col,col))
    ax[0,0].text(L/2,ymid,'h',ha='center',va='bottom',fontsize=25,color=col)

    ax[0,0].set_xlabel('x')
    ax[0,1].set_xlabel('x')
    ax[1,0].set_xlabel('x')
    ax[1,1].set_xlabel('x')

    ax[0,0].set_ylabel(r'$\langle\rho\rangle$')
    ax[0,1].set_ylabel(r'$\langle u_v\rangle$')
    ax[1,0].set_ylabel(r'$\langle u_h\rangle$')
    ax[1,1].set_ylabel(r'$\langle\omega\rangle$')

    ax[0,0].set_xlim(0,L)
    ax[0,1].set_xlim(0,L)
    ax[1,0].set_xlim(0,L)
    ax[1,1].set_xlim(0,L)

    ax[0,0].set_ylim(0,2)

    ax[0,0].grid()
    ax[0,1].grid()
    ax[1,0].grid()
    ax[1,1].grid()

    plt.tight_layout()
    
    if save:
        plt.savefig('figs/SPH_means_{}_N{}k_Nt{}k.png'.format(mode,int(N/1000),int(Nt/1000)),dpi=200)
    plt.show()

def pdfs_graph(y,parameters,save=False):
    #_________________
    #Plot PDFs of uv,ω
    #_________________
    def get_pdf(q,t0,steps=2000,r=(-5,5)):
        ts = list(range(t0,Nt-1))[::steps]
        pdfs = []
        for t in ts:
            x = np.mean(q[t-int(steps/4):t+int(steps/4)+1,:],axis=0)
            hist = np.histogram(x,bins=100,range=r)
            hist = (hist[0]/N,hist[1])
            pdfs.append(hist)
        return pdfs
    
    def plot(pdfs,pdf_type,pdf_type_latex,save):
        gs = grid_spec.GridSpec(len(pdfs),1)
        fig = plt.figure(figsize=(10,13))
        axs = []

        ymax = 0
        for pdf in pdfs:
            ym = max(pdf[0])
            ymax = max(ymax,ym)

        ylims = (-0.1,ymax)

        for i,pdf in enumerate(pdfs):
            axs.append(fig.add_subplot(gs[i:i+1,0:]))
            x= pdf[1][:-1]
            y = pdf[0]
            axs[i].fill(x,y,facecolor='k')
            axs[i].plot(x,y,'-w')
            axs[i].set_ylim(*ylims)

            rect = axs[i].patch
            rect.set_alpha(0)

            spines = ["top","right","left","bottom"]
            for s in spines:
                axs[i].spines[s].set_visible(False)

            axs[i].set_yticklabels([])
            axs[i].set_yticks([])
            axs[i].tick_params(axis='x', colors='white')
            if i != len(pdfs) - 1:
                axs[i].set_xticklabels([])
                axs[i].set_xticks([])
            
        fig.patch.set_facecolor('black')
        fig.text(0.5,0.9,pdf_type_latex,color="white",fontsize=20,ha='center')
        gs.update(hspace=-0.8)
        if save:
            plt.savefig('figs/{}_pdf_{}_N{}k_Nt{}k.png'.format(pdf_type,mode,int(N/1000),int(Nt/1000)),dpi=200)
        plt.show()
    
    Nt,N,dt,mode = parameters

    uv_mean = np.mean(y[0,:,1])
    omega_mean = np.mean(y[0,:,3])
    uv_std = np.sqrt(np.var(y[0,:,1]))
    omega_std = np.sqrt(np.var(y[0,:,3]))

    steps = Nt//50
    uv_pdfs = get_pdf(y[:,:,1],t0=steps//2,steps=steps,r=(uv_mean - uv_std*5,uv_mean + uv_std*5))
    omega_pdfs = get_pdf(y[:,:,3],t0=steps//2,steps=steps,r=(0,omega_mean + omega_std*5))

    plot(uv_pdfs,'uv',r'$u_v$',save)
    plot(omega_pdfs,'omega',r'$\omega$',save)
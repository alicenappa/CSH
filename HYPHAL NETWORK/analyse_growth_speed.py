#Recueil des fonctions ayant pour but de traiter la vitesse de croissance des hyphes
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
from scipy.optimize import curve_fit
from tqdm import tqdm

from Reseau import Reseau, Branche
import fct_utils as utils


def growth_speed_old(reseau:Reseau, tmax:int = 20, vmax:float = 100,filtre = lambda b: True)->list[float]:
    """
    analyse the growth speed of branches
    Return the three parameters of the apical branch fit: V0, Vf and lambda
    """
    bin:float = 5 #width of growth_speed bin iin px/frame
    t_stationary:int = 10
    fig,ax = plt.subplots(figsize=(16,12),
                          nrows=2,ncols=2,
                          gridspec_kw={"height_ratios":[1,2]},
                          constrained_layout=True)
    fig.suptitle(f"{reseau.name}")
    ax[0,0].set_title("Apical branches")
    ax[0,0].sharex(ax[1,0])
    ax[0,1].set_title("Lateral branches")
    ax[0,1].sharex(ax[1,1])
    colors= ["blue","orange"]
    params = np.zeros(shape=(2,2,3))#2 types (Api/Lat)*(3 params + 3 incertitudes)
    for i,case in enumerate(("Apical","Lateral")):
        normes_tot = []
        temps_tot = []
        for b in reseau.branches:
            if b.nature == case and filtre(b):
                positions,vitesses = b.positions_vitesses()
                if len(positions):
                    normes = np.linalg.norm(vitesses,axis=1)
                    temps = positions[:,2]-positions[0,2]
                    normes_tot.append(normes)
                    temps_tot.append(temps)
        normes_tot = np.array([e for l in normes_tot for e in l])
        temps_tot = [e for l in temps_tot for e in l]
        data = {t: np.where(temps_tot==t) for t in set(temps_tot)}
        count = np.zeros(shape=(tmax,int(vmax//bin)),dtype=int)
        vitesses = []
        for t in range(0,tmax):
            vit = [normes_tot[filtre] for i,filtre in data.items() if (t<=i<(t+1))]
            vit = sorted([e for l in vit for e in l if 0<e<vmax])
            for v in vit:
                k = int(v//bin)
                count[t,k] += 1
            vitesses.append(vit)
        normalized = np.array([count[t,:]/sum(count[t,:]) for t in range(0,tmax)]).T
        normalized = normalized[::-1,:]
        temps = np.arange(0,tmax)+.5
        normal_factor = np.sum(count,axis=1)
        ax[0,i].bar(temps,height = normal_factor, width=0.9,
                    color="grey")
        ax[0,i].set_yscale("log")
        ax[0,i].set_ylabel("count")
        ax[1,i].imshow(normalized,cmap="Greys",
                    interpolation="none",extent=[0,tmax,0,vmax],aspect="auto")
        means = [np.mean(vitesses[t]) for t in range(0,tmax)]
        stds = [np.std(vitesses[t]) for t in range(0,tmax)]
        for q in [0.1,0.5,0.9]:
            ax[1,i].plot(temps,[np.quantile(vitesses[t],q) for t in range(tmax)],'k--')
        maxis = [np.argmax(count[t,:])*bin+bin/2 for t in range(tmax)]
        ax[1,i].errorbar(temps,
                        maxis,
                        yerr=stds,
                        color=colors[i])
        ax[1,i].set_ylim([0,vmax])
        ax[1,i].set_xlim([0,tmax])
        ax[1,i].set_ylabel("Growth speed (px/frame)")
        ax[1,i].set_xlabel("Time after initial branching (frame)")
        stationary = [v for t in range(t_stationary,tmax) for v in vitesses[t]]
        hist,bins = np.histogram(stationary,bins = int(vmax//bin))
        imax = np.argmax(hist)
        Vf_i = (bins[imax]+bins[imax+1])*.5
        dVf_i = np.std(stationary)#Nope
        V0_i = maxis[0]
        dV0_i = stds[0] #Nope
        def ajustement(t,V0,Vf,lam):
            return (Vf-(Vf-V0)*np.exp(-lam*t))
        param, covar = curve_fit(ajustement,
                                 temps,#[:t_stationary],
                                 maxis,#[:t_stationary],
                                 sigma=stds,#[:t_stationary],
                                 p0=[V0_i,Vf_i,1],
                                 bounds=([V0_i-1*bin,Vf_i-1*bin,0],[V0_i+1*bin,Vf_i + 1*bin,10]))
        V0,Vf,lam_best = param
        t = np.linspace(0,tmax,100)
        ax[1,i].plot(t,ajustement(t,*param),'r-',
                    label=f"$V_0 =$ {V0:.1f}, $V_f =$ {Vf:.1f}, $\lambda =$ {lam_best:.1f}")
        ax[1,i].legend()
        params[i,0,:] = [V0,Vf,lam_best]
        params[i,1,:] = [np.sqrt(covar[0,0]),np.sqrt(covar[1,1]),np.sqrt(covar[2,2])]
    fig.savefig(reseau.output_dir+"GrowthSpeed_relTime.jpg",dpi=300)
    plt.show()
    plt.close()
    return params

def growth_speed(reseau:Reseau, branches:list[Branche],ax:False, t_max:int=20, v_max:float=100.,label:str="",out=False)->list[float]:
    """
    Estimate V0,Vf and lambda from distribution of growth speed norms from the branches given.
    if ax, show in the ax the distribution
    if out == True, return the list of time and growth speed
    """
    vitesses = {t:[] for t in range(t_max)}
    for b in tqdm(branches):
        pos,vit = b.positions_vitesses()
        pos = pos - pos[0,2]
        normes = np.linalg.norm(vit,axis=1)
        filtre = np.where(np.logical_and(normes>0,
                                         normes<v_max))
        pos = pos[filtre]
        normes = normes[filtre]
        for t,v in zip(pos[:,2],normes):
            if t<t_max:
                vitesses[int(t)].append(v)
    means = np.array([np.mean(vs) for vs in vitesses.values()])
    stds = np.array([np.std(vs) for vs in vitesses.values()])
    normalize = np.array([len(vs) for vs in vitesses.values()])
    times = np.array(list(vitesses.keys()))
    
    def ajust(t,V0,Vf,lambda_):
        return Vf - (Vf-V0)*np.exp(-lambda_*t)
    params = curve_fit(ajust,times+.5,means,sigma=stds/np.sqrt(normalize),p0=[0.,60.,1.])
    print(params)
    if ax:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        tt = [t for t in times for v in vitesses[t]]
        vv = [v for t in times for v in vitesses[t]]
        ww = [1/n for n,t in zip(normalize,times) for v in vitesses[t]]
        hist2d,_,_ = np.histogram2d(tt,vv,bins=[np.arange(0,t_max+1),
                                            np.linspace(0,v_max,20)],
                                weights = ww)
        times = times +.5
        divider = make_axes_locatable(ax)
        axtop = divider.append_axes("top",size="25%", pad="2%")
        fig = ax.get_figure()
        fig.add_axes(axtop)
        axtop.sharex(ax)
        axtop.set_ylabel("count")
        axtop.bar(times,normalize,color="Gray")
        axtop.set_yscale("log")
        ax.text(t_max/2,v_max*.8,label,{"ha":"center"})
        ax.set_ylabel("Growth speed (px/frame)")
        ax.set_xlabel("Time after branch start\n(frame)")
        ax.errorbar(times,means,yerr=stds,color="steelblue")
        tt_smooth = np.linspace(0,t_max,100)
        ax.plot(tt_smooth,ajust(tt_smooth,*params[0]),color="red",linestyle="dashed")
        ax.imshow(hist2d.T[::-1,:],extent=(0,t_max,0,v_max),cmap="Greys",aspect="auto",interpolation=None)
        ax.set_ylim([0,v_max])
    if out:
        return params,vitesses
    return params

def growth_speed_Langevin(reseau:Reseau,branches,V0:float,Vf:float,lambda_:float,v_max:float=100):
    """
    Renvoie FT, FN et F_normes, les projections tangentielles, normales et les normes de la force de Langevin et
    gT, gN, gF les fonctions d'autocorrélation de ces grandeurs.
    """
    FT = []
    FN = []
    F_normes = []
    FiFk_T = []
    FiFk_N = []
    FiFk_normes = []
    for b in branches:
        pos,vit = b.positions_vitesses()
        time = pos[:,2]-pos[0,2]
        normes = np.linalg.norm(vit,axis=1)
        filtre = np.where(np.logical_and(normes>0,
                                         normes<v_max))
        time = time[filtre]
        normes = normes[filtre]
        vit = vit[filtre]
        #On filtre les normes nulles et les normes aberrantes.
        #Il pourrait être interessant de visualiser les normes aberrantes.
        if len(normes)>1:
            #Si on peut calculer des variations, les voici
            dt = time[1:] - time[:-1]
            tt = (time[1:] + time[:-1])/2
            dv = normes[1:] - normes[:-1]
            vv = (normes[1:] + normes[:-1])/2
            FT_current = dv/dt+lambda_*(vv-Vf)
            FT = [*FT,*FT_current]
            #Variations angulaires
            dir_vit = np.zeros_like(vit)
            dir_vit[:,0] = vit[:,0]/normes
            dir_vit[:,1] = vit[:,1]/normes
            dir_vit_orth = np.zeros_like(dir_vit)
            dir_vit_orth[:,0] = -dir_vit[:,1]
            dir_vit_orth[:,1] = +dir_vit[:,0]
            dtheta = np.arctan2(np.sum(dir_vit[1:,:]*dir_vit_orth[:-1,:],axis=1),
                                np.sum(dir_vit[1:,:]*dir_vit[:-1,:],axis=1))
            FN_current = dtheta/dt*vv
            FN = [*FN,*FN_current]
            #Normes:
            F_normes_current = np.sqrt([ft**2+fn**2 for ft,fn in zip(FT_current,FN_current)])
            F_normes = [*F_normes,*F_normes_current]
            #Autocorrelation de la branche
            ti_tk_fifk_T_b = np.array([[t_i,t_k,f_i*f_k] 
                                        for t_i,f_i in zip(tt,FT_current) 
                                        for t_k,f_k in zip(tt,FT_current)])
            ti_tk_fifk_N_b = np.array([[t_i,t_k,f_i*f_k] 
                                        for t_i,f_i in zip(tt,FN_current) 
                                        for t_k,f_k in zip(tt,FN_current)])
            ti_tk_fifk_normes_b = np.array([[t_i,t_k,f_i*f_k]
                                            for t_i,f_i in zip(tt,F_normes_current)
                                            for t_k,f_k in zip(tt,F_normes_current)])
            FiFk_T.append(ti_tk_fifk_T_b)
            FiFk_N.append(ti_tk_fifk_N_b)
            FiFk_normes.append(ti_tk_fifk_normes_b)
    #Applatissements des listes
    FiFk_T = np.array([elem for subl in FiFk_T for elem in subl])
    FiFk_N = np.array([elem for subl in FiFk_N for elem in subl])
    FiFk_normes = np.array([elem for subl in FiFk_normes for elem in subl])
    #Calcul des moyennes pour l'autocorrelation
    g_tau = {}
    for ti,tk,FiFk in FiFk_T:
        deltaT = tk-ti
        liste = g_tau.get(deltaT,[])
        liste.append(FiFk)
        g_tau[deltaT] = liste
    gT = [(tau,np.mean(liste)) for tau,liste in g_tau.items() 
          if tau>=0 and len(liste)>10]
    g_tau = {}
    for ti,tk,FiFk in FiFk_N:
        deltaT = tk-ti
        liste = g_tau.get(deltaT,[])
        liste.append(FiFk)
        g_tau[deltaT] = liste
    gN = [(tau,np.mean(liste)) for tau,liste in g_tau.items() 
          if tau>=0 and len(liste)>10]
    #Calcul des moyennes pour l'autocorrelation
    g_tau = {}
    for ti,tk,FiFk in FiFk_normes:
        deltaT = tk-ti
        liste = g_tau.get(deltaT,[])
        liste.append(FiFk)
        g_tau[deltaT] = liste
    gF = [(tau,np.mean(liste)) for tau,liste in g_tau.items() 
          if tau>=0 and len(liste)>10]
    #Sorting autocorrelation
    gT = np.array(sorted(gT,key=lambda tup:tup[0]))
    gN = np.array(sorted(gN,key=lambda tup:tup[0]))
    gF = np.array(sorted(gF,key=lambda tup:tup[0]))
    return FT,FN,F_normes,gT,gN,gF

def growth_speed_summary(reseau:Reseau,branches,t_max:int=20, v_max:float = 100):
    """
    Calculate V0, Vf, lamdba, FT, FN, autocorrelation function, ks tests for gaussian adequation.
    """
    params, vitesses = growth_speed(reseau, branches, ax=False,
                                    t_max = t_max, v_max = v_max, out=True)
    V0,Vf,lambda_ = params[0]
    lambda_ = 1.1*lambda_
    covar = params[1]
    FT,FN,F_normes,gT,gN,gF = growth_speed_Langevin(reseau,branches,V0,Vf,lambda_,v_max)
    def adjust(t):
        return Vf - (Vf - V0)*np.exp(-lambda_*t)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig,axes = plt.subplot_mosaic([["dist","std","show"],
                                   ["scatter","autocorr","text"]],
                                  figsize=(11.69,8.27),
                                  tight_layout = True)
    fig.suptitle(f"Reseau - {reseau.name}")
    colorFT ="deepskyblue"
    colorFN = "seagreen"
    colorAdjust = "red"
    colorSpeed = "orange"
    #==========================================================================
    divider = make_axes_locatable(axes["dist"])
    axes["count_dist"] = divider.append_axes("top",size="25%", pad="2%")
    axes["count_dist"].text(-0.1, 1.5,"1)", transform=axes["count_dist"].transAxes, 
                            size=20, weight='bold')
    axes["dist"].sharex(axes["count_dist"])
    means = np.array([np.mean(vs) for vs in vitesses.values()])
    stds = np.array([np.std(vs,ddof=1) for vs in vitesses.values()])
    normalize = np.array([len(vs) for vs in vitesses.values()])
    coeffs_c4 = np.array([utils.c_4(n) for n in normalize])
    print(coeffs_c4)
    stds = stds/coeffs_c4
    times = np.array(list(vitesses.keys()))
    tt = [t for t in times for v in vitesses[t]]
    vv = [v for t in times for v in vitesses[t]]
    ww = [1/n for n,t in zip(normalize,times) for v in vitesses[t]]
    hist2d,_,_ = np.histogram2d(tt,vv,bins=[np.arange(0,t_max+1),
                                        np.linspace(0,v_max,20)],
                            weights = ww)
    times = times +.5
    axes["count_dist"].set_ylabel("count")
    axes["count_dist"].bar(times,normalize,color="Gray")
    axes["count_dist"].set_yscale("log")
    axes["dist"].set_ylabel("Growth speed ($px.frame^{-1}$)")
    axes["dist"].set_xlabel("Time after branch start\n(frame)")
    axes["dist"].errorbar(times,means,yerr=stds,color=colorSpeed)
    tt_smooth = np.linspace(0,t_max,100)
    axes["dist"].plot(tt_smooth,adjust(tt_smooth),
                      color=colorAdjust,linestyle="dashed")
    axes["dist"].imshow(hist2d.T[::-1,:],extent=(0,t_max,0,v_max),
                        cmap="Greys",aspect="auto",interpolation=None)
    axes["dist"].set_ylim([0,v_max])
    plt.setp(axes["count_dist"].get_xticklabels(), visible=False)
    #==========================================================================
    axes["std"].text(-0.1, 1.1,"2)", transform=axes["std"].transAxes, 
                     size=20, weight='bold')
    axes["std"].plot(times,stds,color=colorSpeed)
    axes["std"].set_xlabel("Time after branch start\n(frame)")
    axes["std"].set_ylabel("$\sigma$ ($px.frame^{-1}$)")
    axes["std"].set_ylim([0,20])
    D = (gT[0,1]+gN[0,1])/4
    t_smooth = np.linspace(times[0],times[-1],100)
    sigmaF = np.sqrt(D/lambda_)*np.sqrt(1-np.exp(-2*lambda_*t_smooth))
    axes["std"].plot(t_smooth,sigmaF,"k--")
    #==========================================================================
    axes["show"].text(-0.1, 1.1,"3)", transform=axes["show"].transAxes, 
                     size=20, weight='bold')
    axes["show"].imshow(reseau.image_at(reseau.end),
                        cmap="Greys")
    axes["show"].set_axis_off()
    for b in branches:
        axes["show"].plot([reseau.n2x[n] for n in b.noeuds],
                          [reseau.n2y[n] for n in b.noeuds])
    axes["show"].set_aspect("equal")
    #==========================================================================
    from scipy.stats import norm, ks_1samp, ks_2samp
    muT,sigT = np.mean(FT),np.std(FT)
    muN,sigN = np.mean(FN),np.std(FN)
    mu,sig = np.mean([*FT,*FN]),np.std([*FT,*FN])
    gaussian_FT = norm(loc=0,scale=sigN)
    gaussian_FN = norm(loc=0,scale=sigN)
    pvalue_FT = ks_1samp(FT,gaussian_FT.cdf).pvalue
    pvalue_FN = ks_1samp(FN,gaussian_FN.cdf).pvalue
    pvalue_FTFN = ks_2samp(FT,FN).pvalue

    bins = np.linspace(min(muT-3*sigT,muN-3*sigN),
                       max(muT+3*sigT,muN+3*sigN),
                       21)
    divider = make_axes_locatable(axes["scatter"])
    axes["FT"] = divider.append_axes("top",size="20%", pad="2%")
    axes["FT"].text(-0.1, 1.8,"4)", transform=axes["FT"].transAxes, 
                     size=20, weight='bold')
    axes["FN"] = divider.append_axes("right",size="20%", pad="2%")
    axes["scatter"].scatter(FT,FN,marker="x",
                            c=[np.sqrt(ft**2+fn**2)/sigN
                                   for ft,fn in zip(FT,FN)],
                            vmin=0,vmax=4,
                            cmap="RdYlGn_r")
    axes["scatter"].set_xlabel("$F_T (px.frame^{-2})$",color=colorFT)
    axes["scatter"].set_ylabel("$F_N (px.frame^{-2})$",color=colorFN)
    axes["scatter"].sharex(axes["FT"])
    axes["scatter"].sharey(axes["FN"])
    axes["scatter"].hlines(0,-3*sigN,3*sigN,
                           colors="black",linestyles="dashed")
    axes["scatter"].vlines(0,-3*sigN,3*sigN,
                           colors="black",linestyles="dashed")
    axes["scatter"].set_xlim([-4*sigN,+4*sigN])
    axes["scatter"].set_ylim([-4*sigN,+4*sigN])
    axes["scatter"].set_aspect('equal')
    thetas = np.linspace(0,2*np.pi,100)
    xx = sigN*np.cos(thetas)
    yy = sigN*np.sin(thetas)
    for r in range(1,4):
        axes["scatter"].plot(r*xx,r*yy,alpha=1/r,color="grey",ls="--")
    axes["FT"].hist(FT,color=colorFT,
                    bins=bins,density=True)
    axes["FN"].hist(FN,color=colorFN,
                    bins=bins,density=True,orientation="horizontal")
    axes["FT"].tick_params(axis="x", labelbottom=False)
    axes["FN"].tick_params(axis="y", labelleft=False)
    axes["FT"].set_ylabel("density")
    axes["FN"].set_xlabel("density")
    xx_FT = np.linspace(muT-3*sigT,muT+3*sigT,100)
    xx_FN = np.linspace(muN-3*sigN,muN+3*sigN,100)
    axes["FT"].plot(xx_FT,gaussian_FT.pdf(xx_FT),'k--')
    axes["FN"].plot(gaussian_FN.pdf(xx_FN),xx_FN,'k--')
    #==========================================================================
    axes["autocorr"].text(-0.1, 1.1,"5)", transform=axes["autocorr"].transAxes, 
                          size=20, weight='bold')
    axes["autocorr"].plot(gT[:,0],gT[:,1],color=colorFT,marker="o")
    axes["autocorr"].stem(gT[:,0],gT[:,1],linefmt=colorFT,markerfmt=colorFT)
    axes["autocorr"].plot(gN[:,0],gN[:,1],color=colorFN,marker="o")
    axes["autocorr"].stem(gN[:,0],gN[:,1],linefmt=colorFN,markerfmt=colorFN)
    #axes["autocorr"].plot(gF[:,0],gF[:,1]-np.mean(F_normes)**2)
    #axes["autocorr"].stem(gF[:,0],gF[:,1]-np.mean(F_normes)**2)
    axes["autocorr"].set_xlim([-0.5,t_max+.5])
    axes["autocorr"].set_xlabel("Time lag $\\tau$ ($frame$)")
    axes["autocorr"].set_ylabel("$<F(t)F(t-\\tau)>$ ($px^2.frame^{-4}$)")
    #==========================================================================
    axes["text"].text(-0.1, 1.1,"6)", transform=axes["text"].transAxes, 
                     size=20, weight='bold')
    axes["text"].set_axis_off()
    axes["text"].text(0,3,f"$v_0 = {V0:.2f}\pm{np.sqrt(covar[0][0]):.2f}$")
    axes["text"].text(0,2,f"$v_f = {Vf:.2f}\pm{np.sqrt(covar[1][1]):.2f}$")
    axes["text"].text(0,1,f"$\lambda = {lambda_:.2f}\pm{np.sqrt(covar[2][2]):.2f}$")
    axes["text"].text(0,0,f"$D = {D:.2f} \pm {np.abs(gT[0,1]-gN[0,1])/4:.2f}$")
    axes["text"].text(0,-1,f"$\mu_T = {muT:.2f}$\t$\sigma_T = {sigT:.2f}$",color=colorFT)
    axes["text"].text(0,-2,f"$\mu_N = {muN:.2f}$\t$\sigma_N = {sigN:.2f}$",color=colorFN)
    axes["text"].text(0,-3,"Test Kolmogorov-Smirnov :")
    axes["text"].text(0,-4,"$F_T$ vs $N(0,\sigma_N)$ p-value = "+f"{pvalue_FT:.1e}",color=colorFT)
    axes["text"].text(0,-5,"$F_N$ vs $N(0,\sigma_N)$ p-value = "+f"{pvalue_FN:.1e}",color=colorFN)
    axes["text"].text(0,-6,f"$F_T$ vs $F_N$ p-value = {pvalue_FTFN:.1e}")
    axes["text"].set_ylim([-6,4])
    fig.tight_layout()
    fig.savefig(reseau.output_dir+f"{reseau.name}_growth_speed.jpg",dpi=300)
    plt.show()
    plt.close()

def visu_slow_branches(reseau:Reseau,branches,tseuil:int=5,Vseuil:float = 40):
    """
    Visualisation des branches lentes afin d'identifier les potentiels problèmes permettant
    d'expliquer la surestimation de l'écart type sigma et la sous estimation de vf.
    """
    for b in branches:
        pos,vit = b.positions_vitesses()
        if len(vit)>tseuil:
            normes = np.linalg.norm(vit,axis=1)
            if np.any(normes[tseuil:]<Vseuil):
                fig,(ax1,ax2) = plt.subplots(figsize=(8,8),nrows=1,ncols=2)
                trel =pos[:,2] - pos[0,2]
                colors = ["orange" if (n>Vseuil or t<tseuil) else "red" 
                          for n,t in zip(normes,trel)]
                ax2.quiver(pos[:,0],pos[:,1],vit[:,0],vit[:,1],
                           color=colors,pivot="mid",
                           angles="xy",scale_units="xy",scale=1)
                xlim = ax2.set_xlim()
                ylim = ax2.set_ylim()
                ax2.imshow(Image.open(reseau.imgs.format(int(np.max(pos[:,2]))+1)),
                           cmap="Greys")
                ax2.set_xlim(xlim)
                ax2.set_ylim(ylim)
                ax1.plot(trel,normes)
                ax1.scatter(trel,normes,marker="x",color=colors)
                ax1.plot((trel[1:]+trel[:-1])/2,(normes[1:]+normes[:-1])/2,'k--')
                ax1.set_ylim([0,100])
                ax1.hlines(Vseuil,0,trel[-1],colors="red",linestyles="dashed")
                ax1.set_xlabel("Temps de croissance")
                ax1.set_ylabel("Vitesse de croissance")
                plt.draw()
                plt.pause(0.01)
                input("Press enter to continue")
                plt.close()

def growth_speed_noise(reseau:Reseau,branches,ax,V0:float,Vf:float,lambda_:float,v_max:float=100):
    """
    Teste l'adéquation de la force de Langevin de la note "GrowthSpeed" sur les branches données avec les 
    paramètres préalablement calculés.
    """
    FT = []#Projection tangentielle de la force de Langevin
    FN = []#Projection normale de la force de Langevin
    times = []#A quel instant correspondent ces projections. Utile pour l'autocorrélation peut être.
    FiFk_T = [] #g(tau) = <F(t)F(t-tau)> Les 2 F doivent correspondre à la même branche.
    FiFk_N = []
    #La moyenne au dessus correspond à la moyenne sur la branche pour un décalage de tau.
    def v_adjust(t):
        """
        Fonction temporaire d'ajustement de la vitesse de croissance.
        Non utilisé maintenant...
        """
        return Vf-(Vf-V0)*np.exp(-lambda_*t)
    for b in branches:
        pos,vit = b.positions_vitesses()
        time = pos[:,2]-pos[0,2]
        normes = np.linalg.norm(vit,axis=1)
        filtre = np.where(np.logical_and(normes>0,
                                         normes<v_max))
        time = time[filtre]
        normes = normes[filtre]
        vit = vit[filtre]
        #On filtre les normes nulles et les normes aberrantes.
        #Il pourrait être interessant de visualiser les normes aberrantes.
        if len(normes)>1:
            #Si on peut calculer des variations, les voici
            dt = time[1:] - time[:-1]
            tt = (time[1:] + time[:-1])/2
            dv = normes[1:] - normes[:-1]
            vv = (normes[1:] + normes[:-1])/2
            FT_current = dv/dt+lambda_*(vv-Vf)
            FT = [*FT,*FT_current]
            #Variations angulaires
            dir_vit = np.zeros_like(vit)
            dir_vit[:,0] = vit[:,0]/normes
            dir_vit[:,1] = vit[:,1]/normes
            dir_vit_orth = np.zeros_like(dir_vit)
            dir_vit_orth[:,0] = -dir_vit[:,1]
            dir_vit_orth[:,1] = +dir_vit[:,0]
            dtheta = np.arctan2(np.sum(dir_vit[1:,:]*dir_vit_orth[:-1,:],axis=1),
                                np.sum(dir_vit[1:,:]*dir_vit[:-1,:],axis=1))
            FN_current = dtheta/dt*vv
            FN = [*FN,*FN_current]
            #Temps
            times = [*times,*tt]
            #Autocorrelation de la branche
            ti_tk_fifk_T_b = np.array([[t_i,t_k,f_i*f_k] 
                                        for t_i,f_i in zip(tt,FT_current) 
                                        for t_k,f_k in zip(tt,FT_current)])
            ti_tk_fifk_N_b = np.array([[t_i,t_k,f_i*f_k] 
                                        for t_i,f_i in zip(tt,FN_current) 
                                        for t_k,f_k in zip(tt,FN_current)])
            FiFk_T.append(ti_tk_fifk_T_b)
            FiFk_N.append(ti_tk_fifk_N_b)
    FiFk_T = np.array([elem for subl in FiFk_T for elem in subl])
    FiFk_N = np.array([elem for subl in FiFk_N for elem in subl])
    g_tau = {}
    for ti,tk,FiFk in FiFk_T:
        deltaT = tk-ti
        liste = g_tau.get(deltaT,[])
        liste.append(FiFk)
        g_tau[deltaT] = liste
    autocorr = []
    for tau,liste in g_tau.items():
        autocorr.append([tau,np.mean(liste)]) 
    autocorr = np.array(autocorr)
    plt.figure()
    plt.stem(autocorr[:,0],autocorr[:,1])
    plt.xlim([-0.5,30.5])
    plt.xlabel("time lag (frame)")
    plt.show()
    plt.close()
    #Regroupement par instant correspondant ?
    times = np.array(times)
    muT,sigT = np.mean(FT),np.std(FT)
    muN,sigN = np.mean(FN),np.std(FN)
    N = len(FT)
    from scipy.stats import norm, ks_1samp, ks_2samp
    gaussian_FT = norm(loc=muT,scale=sigT)
    gaussian_FN = norm(loc=muN,scale=sigN)
    pvalue_FT = ks_1samp(FT,gaussian_FT.cdf).pvalue
    pvalue_FN = ks_1samp(FN,gaussian_FN.cdf).pvalue
    pvalue_FTFN = ks_2samp(FT,FN).pvalue
    pvalues = [pvalue_FT,pvalue_FN,pvalue_FTFN]
    print("Tests Kolmogorov-Smirnov :")
    print(f"Projection tangentielle et Gaussienne : {pvalue_FT:.3f}")
    print(f"Projection normale et Gaussienne : {pvalue_FN:.3f}")
    print(f"Projection tangentielle et normale : {pvalue_FTFN:.3f}")
    xx_FT = np.linspace(muT-3*sigT,muT+3*sigT,1000)
    xx_FN = np.linspace(muN-3*sigN,muN+3*sigN,1000)
    cumulative_y = np.cumsum([1/N for _ in FT])
    cumulative_FT = sorted(FT)
    cumulative_FN = sorted(FN)
    ax.plot(cumulative_FT,cumulative_y,color="midnightblue",label="$F_T$")
    ax.plot(xx_FT,gaussian_FT.cdf(xx_FT),color="deepskyblue",linestyle="dashed",label=f"Gaussian $({muT:.1f},{sigT:.1f})$")
    ax.plot(cumulative_FN,cumulative_y,color="darkgreen",label="$F_N$")
    ax.plot(xx_FN,gaussian_FN.cdf(xx_FN),color="limegreen",linestyle="dashed",label=f"Gaussian $({muN:.1f},{sigN:.1f})$")
    ax.set_xlabel("Projection de F ($px/frame^2$)")
    ax.set_ylabel("Fonction de répartition")
    ax.set_xlim([min(xx_FT[0],xx_FN[0]),max(xx_FT[-1],xx_FN[-1])])
    ax.legend()
    return pvalues,FT,FN

def vitesses_correlation_Nicois(reseau:Reseau, cas:str = "All"):
    """
    WIP
    Renvoie la liste des vecteurs vitesses comme demandé par les matheux 
    de Nice.
    Pour chaque branche considérée:
    On calcule V(n+3/2)-V(n+1/2)
    On projette sur t = unit(V(n+3/2)+V(n+1/2)) et n = t_orth
    L'argument 'cas' permet de choisir si l'on considère toutes les 
    branches, que les branches apicales ou que les branches latérales.
    """
    dX,dY,dT = reseau.n2x,reseau.n2y,reseau.n2t
    projections_t = []
    projections_n = []
    branches_considerees = {"All": reseau.branches,
                            "Apicales":(b for b in reseau.branches 
                                        if b.nature == "Apical"),
                            "Laterales":(b for b in reseau.branches
                                         if b.nature == "Lateral")}
    branches = branches_considerees.get(cas,False)
    if not branches:
        raise ValueError("Cas non valide, merci de choisir parmi "+\
                            "'All', 'Apicales' ou 'Laterales")
    for b in branches:
        positions, vec_vitesses = b.positions_vitesses()
        if len(vec_vitesses>1):
            accelerations = (vec_vitesses[1:,:]-vec_vitesses[:-1,:])
            times = positions[:,2]
            accelerations[:,0] /= (times[1:]-times[:-1])
            accelerations[:,1] /= (times[1:]-times[:-1])
            tangentes = vec_vitesses[:-1,:]+vec_vitesses[1:,:]
            normes = np.linalg.norm(tangentes,axis=1)
            normes = np.where(normes == 0, 1, normes)
            tangentes[:,0] /= normes
            tangentes[:,1] /= normes
            normales = tangentes[:,[1,0]]
            normales[:,0] = - normales[:,0] 
            projT = np.sum(accelerations*tangentes,axis=1)
            projN = np.sum(accelerations*normales,axis=1)
            projections_t.append(projT)
            projections_n.append(projN)
    return projections_t,projections_n

def growth_speed_multireseaux(reseaux:list[Reseau],branches:list[list[Branche]]):
    """
    Analyse plusieurs réseaux à la fois et les compare.
    Pensé pour traiter un triplicat
    """
    t_max = 15
    v_max = 120
    colors = ["skyblue","teal","royalblue","firebrick"]
    offset = [-.3,-.1,.1,.3]
    labels = [r.name for r in reseaux]
    labels.append("All")
    vitesses_tot = {t:[] for t in range(t_max)}

    def ajust(t,V0,Vf,lambda_):
        """
        Loi que doit suivre l'ajustement
        """
        return Vf - (Vf-V0)*np.exp(-lambda_*t)
    params = [] #Collect the parameters and the covarmatrix for each reseau
    means = []
    stds = []
    for k,brs in enumerate(branches):
        vitesses = {t:[] for t in range(t_max)}
        for branch in brs:
            pos,vit = branch.positions_vitesses()
            pos = pos - pos[0,2]
            normes = np.linalg.norm(vit,axis=1)
            filtre = np.where(np.logical_and(normes>0,
                                             normes<v_max))
            pos = pos[filtre]
            normes = normes[filtre]
            for t,v in zip(pos[:,2],normes):
                if t<t_max:
                    vitesses[int(t)].append(v)
                    vitesses_tot[int(t)].append(v)
        means.append(np.array([np.mean(vs) for vs in vitesses.values()]))
        stds.append(np.array([np.std(vs) for vs in vitesses.values()]))
        normalize = np.array([len(vs) for vs in vitesses.values()])
        times = np.array(list(vitesses.keys()))
        params.append(curve_fit(ajust,times+.5,means[-1],sigma=stds[k]/np.sqrt(normalize),p0=[0.,60.,1.]))
    means.append(np.array([np.mean(vs) for vs in vitesses_tot.values()]))
    stds.append(np.array([np.std(vs) for vs in vitesses_tot.values()]))
    normalize = np.array([len(vs) for vs in vitesses_tot.values()])
    times = np.array(list(vitesses_tot.keys()))
    params.append(curve_fit(ajust,times+.5,means[-1],sigma=stds[k]/np.sqrt(normalize),p0=[0.,60.,1.]))
    fig,ax = plt.subplots(figsize=(11.69,8.27))
    
    tt = [t for t in times for v in vitesses_tot[t]]
    vv = [v for t in times for v in vitesses_tot[t]]
    ww = [1/n for n,t in zip(normalize,times) for v in vitesses_tot[t]]
    hist2d,_,_ = np.histogram2d(tt,vv,bins=[np.arange(0,t_max+1),
                                        np.linspace(0,v_max,20)],
                                weights = ww)
    times = times +.5
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    axtop = divider.append_axes("top",size="25%", pad="2%")
    fig.add_axes(axtop)
    axtop.sharex(ax)
    axtop.set_ylabel("count")
    axtop.bar(times,normalize,color=colors[-1])
    axtop.set_yscale("log")
    axtop.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("Growth speed (px/frame)")
    ax.set_xlabel("Time after branch start\n(frame)")
    tt_smooth = np.linspace(0,t_max,100)
    for k,label in enumerate(labels):
        ms = means[k]
        ss = stds[k]
        color = colors[k]
        off = offset[k]
        param = params[k]
        ax.errorbar(times+off,ms,yerr=ss,color=color,marker="x",label=label,
                    xerr=[[.5+off for _ in ms],[+.5-off for _ in ms]])
        ax.plot(tt_smooth,ajust(tt_smooth,*param[0]),color=color,linestyle="dashed",label=f"{label} fit")
    ax.imshow(hist2d.T[::-1,:],extent=(0,t_max,0,v_max),cmap="Greys",aspect="auto",interpolation=None)
    ax.set_ylim([0,v_max])
    ax.legend()
    plt.show()
    plt.close()

    fig,ax = plt.subplots(figsize=(8,8))
    ax2 = ax.twinx()
    V0 = [param[0][0] for param in params]
    dV0 = np.sqrt([param[1][0,0] for param in params])
    Vf = [param[0][1] for param in params]
    dVf = np.sqrt([param[1][1,1] for param in params])
    lam = [param[0][2] for param in params]
    dlam = np.sqrt([param[1][2,2] for param in params])
    for k in range(len(V0)):
        off = offset[k]
        ax.errorbar(0+off,V0[k],yerr=dV0[k],color=colors[k],marker="+")
        ax.errorbar(1+off,Vf[k],yerr=dVf[k],color=colors[k],marker="+")
        ax2.errorbar(2+off,lam[k],yerr=dlam[k],color=colors[k],marker="+")
    ax.set_xticks([0,1,2],labels=["$v_0$", "$v_f$", "$\lambda$"])
    ax.set_ylabel("Speed ($px.frame^{-1}$)")
    ax2.set_ylabel("Rate ($frame^{-1}$)")
    ax.set_ylim([20,65])
    ax.vlines(1.5,0,70,colors="black",linestyles="dashed")
    plt.show()
    plt.close()
    return params
    

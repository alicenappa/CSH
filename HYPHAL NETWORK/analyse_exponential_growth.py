#Imports needed
import numpy as np #For everything                              version 
import matplotlib.pyplot as plt #For visualisation              version 
import networkx as nx #For the graph                            version 
from pickle import load as pload #For loading the gpickle file  version 
from PIL import Image #For image management                     version 
from tqdm import tqdm #For nice loading bars                    version 

#Our part:
from Reseau import Reseau
from declaration import load

def croissance_exponentielle(reseaux:list[Reseau])->list[float]:
    """
    Calcule le coefficient lambda de croissance exponentielle des reseaux dans les
    cas où l'on regarde tous les noeuds de degré 3 où uniquement ceux correspondant à un branchement.
    """

    def lambda_croissance_expo(reseau:Reseau):
        """
        Ajustement du nombre de noeuds de degré 3 au cours du temps
        selon une loi de croissance exponentielle.
        return the times tt, the number of d3 at each time Nb_d3 and 
        the fitted parameters lambda_ and C0 so that Nb_d3(tt)=C0*exp(lambda_*tt)
        """
        tt = np.arange(reseau.start,reseau.end+1)
        Nb_d3 = np.zeros_like(tt)
        for k,t in tqdm(enumerate(tt)):
            g = reseau.network_at(t)
            Nb_d3[k] = sum(d==3 for n,d in g.degree())
        ln_Nb_d3 = np.log(Nb_d3)
        lambda_,c0 = np.polyfit(tt,ln_Nb_d3,deg=1)
        C0 = np.exp(c0)
        return tt,Nb_d3,C0,lambda_
    
    def lambda_branchements_expo(reseau:Reseau):
        """
        Ajustement du nombre de noeuds de degré 3 au cours du temps
        selon une loi de croissance exponentielle.
        return the times tt, the number of d3 at each time Nb_d3 and 
        the fitted parameters lambda_ and C0 so that Nb_d3(tt)=C0*exp(lambda_*tt)
        """
        tt = np.arange(reseau.start,reseau.end+1)
        t_branching = [b.get_tstart() for b in reseau.branches]
        Nb_branching = [sum(tstart<=t for tstart in t_branching)
                        for t in tt]
        ln_Nb_branching = np.log(Nb_branching)
        lambda_,c0 = np.polyfit(tt,ln_Nb_branching,deg=1)
        C0 = np.exp(c0)
        return tt,Nb_branching,C0,lambda_
    colors = ["seagreen","purple","steelblue",
              "darkorange","red","blue",
              "green","black"]
    NColors = len(colors)
    params = [[],[]]
    fig,ax = plt.subplots(figsize=(12,8))
    fig.suptitle("Tout degree 3")
    for k,r in enumerate(reseaux):
        tt,Nb_d3,C0,lambda_ = lambda_croissance_expo(r)
        params[0].append((C0,lambda_))
        ax.scatter(tt,Nb_d3,marker='+',color=colors[k%NColors])
        ax.plot(tt,C0*np.exp(lambda_*tt),color=colors[k%NColors],ls="--",
                label=f"$\lambda = {lambda_:.2f} (frame^{-1}$) - {r.name}")
    ax.set_yscale("log")
    ax.set_xlabel("time (frame)")
    ax.set_ylabel("Degree 3 nodes (count)")
    ax.legend()
    plt.savefig("/Users/thibault/Documents/Thèse/AnalyseReseau/croissance_exponentielle_lambda_NObranchements.png",dpi=300)
    plt.show()
    plt.close()
    fig,ax = plt.subplots(figsize=(12,8))
    fig.suptitle("Branchements")
    for k,r in enumerate(reseaux):
        tt,Nb_d3,C0,lambda_ = lambda_branchements_expo(r)
        params[1].append((C0,lambda_))
        ax.scatter(tt,Nb_d3,marker='+',color=colors[k%NColors])
        ax.plot(tt,C0*np.exp(lambda_*tt),color=colors[k%NColors],ls="--",
                label=f"$\lambda = {lambda_:.2f} (frame^{-1}$) - {r.name}")
    ax.set_yscale("log")
    ax.set_xlabel("time (frame)")
    ax.set_ylabel("Degree 3 nodes (count)")
    ax.legend()
    plt.savefig("/Users/thibault/Documents/Thèse/AnalyseReseau/croissance_exponentielle_lambda_branchements.png",dpi=300)
    plt.show()
    plt.close()
    return params

def main():
    reseaux_names = [
        ("M2WT_200616","overlaps"),
        ("M2WT_200617","overlaps"),
        ("M0WT_200615","overlaps"),
        ("KCl_210111","overlaps"),
        ("T35_220127","overlaps"),
        ("T35_220309","overlaps")
    ]
    reseaux = [load(name,suffix) for name,suffix in reseaux_names]
    print(croissance_exponentielle(reseaux))
    return 1

if __name__=="__main__":
    main()
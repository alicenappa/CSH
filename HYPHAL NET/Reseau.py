"""
Ce fichier contient les définitions des classes Réseau, Brindille et Branche.
Ces trois classes ont été développées pour analyser les réseaux de champignons
de Podospora anserina en définissant de manière unique un graphe dynamique pour
chaque expérience.
Pour plus de détail, se référer à l'article :
"Full identification of a growing and branching network's spatio-temporal 
structures", T. Chassereau, F. Chapeland-Leclerc et E. Herbert, 2024-25

Documentation mise à jour en novembre 2024 par T. Chassereau
Contact possible : thibault.chassereau@gmail.com
"""


import numpy as np #For everything
import matplotlib.pyplot as plt #For visualisation
import networkx as nx #For graph object
from tqdm import tqdm #For nice loading bar
import os  #For creating folder
import pickle #For saving object
from PIL import Image #To load experiment images

#Homemade imports
import fct_utils as fct


class Reseau():
    """
    Définition de la classe Reseau.
    
    Chaque instance de cette classe regroupe l'ensemble des informations 
    nécessaires à l'analyse d'une expérience à savoir l'ensemble des images
    en niveaux de gris au cours du temps, le temps de début d'analyse et
    d'arrêt, les deux images binarisées correspondantes et le graphe spatial
    associé à la dernière image.
    Pour plus d'information sur la génération du graphe spatial, se 
    référer aux fichiers 'TotalReconstruction.py' et 'Vectorisation.py'.
    """
    #float|int : Length treshold between 2 nodes over which it is considered false
    SEUIL_LONG:float = 10 # in hyphal diameter unit
    #int : Seuil de "latence" avant classification en branche latérale
    SEUIL_LAT:int = 5 #frames
    #float|int : Seuil longueur départ de branche
    SEUIL_DEPART:float = 2 # in hyphal diameter unit
    #int : Seuil nombre de boucle lors du calcul de la dynamique 
    #      pour considérer un cas suspect.
    SEUIL_BOUCLE_DYNAMIQUE:int = 40

    """
    ===========================================================================
    Declaration et representation
    """
    def __init__(self,
                 name:str, #str, name of the experiment
                 g:nx.Graph, #Networkx graph of the experiment
                 imgs:dict[int,str], #{frame: img_path} Sequence of images
                 manip_dir:str, #str, Experiment's folder
                 output_dir:str, #str, Output folder
                 first, #First image Binarized
                 last, #Last image Binarized
                 start:int,end:int, #ints, idx of start, and end of experiment
                 brindilles:list = None,#Brindilles
                 branches:list = None,#Branches
                 diameter:float = 7#float, Hyphal diamater in pixels
                 ):
        """
        Définit l'initialisation d'une instance de Réseau.
        """
        self.name = name
        self.g = g
        self.imgs = imgs
        self.first = first
        self.last = last
        self.manip_dir = manip_dir
        self.output_dir = output_dir
        self.start = start
        self.end = end
        #Initialisation of some characteristic of the Reseau
        self.brindilles = brindilles if brindilles is not None else [] #List of twigs
        self.branches = branches if branches is not None else []#List of Branches
        self.source = None
        self.diameter = diameter
        self.n2x = nx.get_node_attributes(g,"x")
        self.n2y = nx.get_node_attributes(g,"y")
        self.n2t = {}
        #Initialisation des dossiers de sortie
        #-> Il  y a du ménage à faire par ici
        directories = ["", "Graphes","GraphesVisu","Overlay",
                       "PotentialErrors","Binarization","Branches",
                       "Vitesses"]
        for directory in directories:
            if not os.path.exists(output_dir+directory):
                os.mkdir(output_dir+directory)

    def __repr__(self) -> str:
        """
        Définit ce qui s'affiche lorsque qu'on utilise 'print' avec le réseau
        comme argument.
        """
        repr = "\n"
        repr += "-"*80
        repr += f"\nReseau {self.name}\n"
        repr += f"\tStart frame {self.start}\n"
        repr += f"\tEnd frame {self.end}\n"
        repr += "-"*80
        repr += "\n"
        return repr
    
    """
    ===========================================================================
    Filtres et correctifs
    """
    def filtres(self):
        """
        Applique les différents filtres au réseau.
        """
        print()
        print("-"*80)
        print("\tDébut du nettoyage")
        self.filtre_longueur()
        self.filtre_doublon()
        encore = True
        while(encore):
            l = 0
            l += self.filtre_depart()
            #l += self.filtre_boucle()
            #l += self.filtre_oeillets()
            encore = l != 0

        print("\tFin du nettoyage")
        print("-"*80)
        print()
        return 1

    def filtre_doublon(self):
        """
        Filtre les noeuds doublons (exact même position) du réseau.
        """
        pos = np.array([[n,self.n2x[n],self.n2y[n]] for n in self.g.nodes])
        #On crée un pointeur qui indique la nouvelle destination des noeuds à
        #supprimer afin de garder un réseau entièrement connecté.
        pointeur = {n:n for n in self.g.nodes}
        for k,[n,x,y] in tqdm(enumerate(pos[:-1]),
                              desc="Recherche des doublons..."):
             if pointeur[n]==n:
                 js = np.argwhere((pos[k+1:,1]==x)&
                                  (pos[k+1:,2]==y)).flatten()+k+1
                 for j in js:
                     pointeur[pos[j,0]] = n
        #Gain d'un facteur 10 en temps de calcul par rapport au calcul de la
        #distance des noeuds.
        h = self.g.copy()
        weights = nx.get_edge_attributes(self.g,"weight")
        #Suppression des noeuds en trop
        toDelete = [n for n in self.g.nodes if pointeur[n]!=n]
        h.remove_nodes_from(toDelete)
        #Reconstruction des liens nécessaires
        for (u,v) in self.g.edges:
            if (u,v) not in h.edges:
                u_eff = pointeur[u]
                v_eff = pointeur[v]
                h.add_edge(u_eff,v_eff)
                nx.set_edge_attributes(h,
                                       {(u_eff,v_eff):weights[(u,v)]},"weight")
        print(f"Nombre de noeuds supprimés : {len(toDelete)}")
        print(f"Soit : {len(toDelete)/len(self.g.nodes)*100:.2f}% des noeuds.")
        print("Petit nettoyage supplémentaire pour bien finir ...")
        self.g = h
        self.filtre_autoconnection()
        return 1

    def filtre_autoconnection(self):
        """
        Filtre les autoconnections (lien (u,v) avec u==v) du réseau.
        """
        h = self.g.copy()
        c = 0
        for (u,v) in tqdm(self.g.edges,desc="Filtrage des autoconnections..."):
            if u==v:
                h.remove_edge(u,v)
                c += 1
        self.g = h
        print(f"Suppression de {c} autoconnection(s)")
        return 0

    def filtre_boucle(self):
        """
        Filtre les boucles (brindille ayant le même noeud pour début et fin)
        du réseau.
        """
        h = self.g.copy()
        c = 0
        h_prune = fct.prune(h)
        for n in (n for n,d in h_prune.degree() if d==2):
            if h.degree(n) == 3:
                c += 1
                #A la recherche de la boucle.
                [v1,v2,v3] = list(h.neighbors(n))
                #pose problème...
                v1v2 = sorted(list(nx.all_simple_paths(h,
                                                       source = v1,
                                                       target = v2,
                                                       cutoff = 100)),
                              key = len)[1]
                v1v3 = sorted(list(nx.all_simple_paths(h,
                                                       source = v1,
                                                       target = v3,
                                                       cutoff = 100)),
                              key = len)[1]
                v2v3 = sorted(list(nx.all_simple_paths(h,
                                                       source = v2,
                                                       target = v3,
                                                       cutoff = 100)),
                              key = len)[1]
                boucle = sorted([v1v2,v1v3,v2v3],key=len)[0]
                if np.any(np.array([d==3 for n,d in h.degree(boucle)])):
                    continue
                h.remove_nodes_from(boucle)
        print(f"{c} boucle(s) enlevée(s)")
        self.g = h
        """
        #Visualisation de contrôle...
        dX = nx.get_node_attributes(h,"x")
        dY = nx.get_node_attributes(h,"y")
        d2 = (n for n,d in h_prune.degree() if d == 2)
        fig,ax = plt.subplots()
        show(h)
        for n in d2:
            ax.scatter(dX[n],dY[n],color="red",marker="o")
        plt.show()
        plt.close(fig)
        """
        return c

    def filtre_longueur(self):
        """
        Filtre les liens du réseau extrèmement longs.
        Plus long que Reseau.SEUIL_LONG.
        Renvoie la longueur totale des liens supprimés.
        """

        h = self.g.copy()
        toDelete = []
        Seuil2 = (Reseau.SEUIL_LONG*self.diameter)**2
        c = 0
        longTot = 0
        for (u,v) in tqdm(h.edges,desc= "Filtration des liens trop long ..."):
            L2 = (self.n2x[u]-self.n2x[v])**2+(self.n2y[u]-self.n2y[v])**2
            if L2 > Seuil2:
                toDelete.append((u,v))
                longTot += np.sqrt(L2)
                c += 1
        h.remove_edges_from(toDelete)
        h = h.subgraph(sorted(nx.connected_components(h),
                              key=len,reverse=True)[0]).copy()
        self.g = h
        print(f"{c} lien(s) supprimé(s) car trop long.")
        return longTot

    def filtre_oeillets(self)->int:
        """
        Filtre les oeillets (deux brindilles reliant les mêmes extrémités).
        Ils sont généralement dus à une impureté sur le bord de l'hyphe.
        Ils forment des d2 lors de l'élagage, ce qui peut poser problème.
        """
        h = self.g.copy()
        h_prune = fct.prune(h)
        toDelete = []
        c = 0
        d2 = [n for n,d in h_prune.degree() if d == 2]
        for k,u in enumerate(d2):
            for v in h_prune.neighbors(u):
                if v in d2[k:]:
                    #u et v sont les deux extrémités d'un oeillet...
                    #Il faut donc trouver les deux chemins reliant ces noeuds
                    paths = nx.all_simple_paths(h,
                                                source = u,
                                                target = v,
                                                cutoff = 100)
                    paths = sorted(list(paths),key= len)[:2]
                    #On supprime le chemin le plus long de l'oeillet ?
                    c += 1
                    toDelete = [*toDelete,*(paths[1][1:-1])]
        h.remove_nodes_from(toDelete)
        print(f"{c} oeillet(s) supprimé(s)")
        self.g = h
        """
        #Visualisation de contrôle...
        hist = [d for n,d in prune(h).degree()]
        fig,ax = plt.subplots()
        ax.hist(hist)
        plt.show()
        plt.close(fig)
        dX = nx.get_node_attributes(h,"x")
        dY = nx.get_node_attributes(h,"y")
        fig,ax = plt.subplots()
        show(h)
        for u in d2:
            plt.scatter(dX[u],dY[u],color="red",marker="x")
        plt.show()
        plt.close(fig)
        """
        return c

    def filtre_depart(self)->float:
        """
        Filtre les brindilles terminales trop courtes.
        Si distance Apex - d3 est < SEUIL_DEPART alors on coupe.
        Ne nécessite pas de datation des noeuds. S'applique sur le réseau
        à la dernière frame considérée.
        """
        h = self.g.copy()
        d1 = [n for n,d in self.g.degree if d == 1]
        d3 = [n for n,d in self.g.degree if d == 3]
        c = 0
        h_prune = fct.prune(h)
        long_supp = 0
        for a in tqdm(d1,desc="Verification de la longueur des départs ... "):
            for d3 in h_prune.neighbors(a):
                chemin = nx.shortest_path(h,source=a,target=d3)
                pos = np.array([[self.n2x[n],self.n2y[n]] for n in chemin])
                L = np.sum(np.sqrt(((pos[1:,:]-pos[:-1,:])**2).sum(axis=1)))
                if L < Reseau.SEUIL_DEPART*self.diameter:
                    h.remove_nodes_from(chemin[:-1])
                    long_supp += L
                    c+=len(chemin)-1
        print(f"{c} noeud(s) supprimé(s) pour cause de départ précoce")
        self.g = h
        return long_supp

    def correctif_depart(self):
        """
        Corrige les départs de brindilles pour qu'à aucune frame il 
        n'existe pas de brindille terminale de longueur 
        inférieure à SEUIL_DEPART.
        Protection du graphe initial qui ne doit pas être modifié.
        """
        g0 = self.network_at(self.start)
        for f in tqdm(range(self.start+1,self.end),
                      desc = "Correction des départs précoces ... "):
            g = self.network_at(f)
            for apex in (n for n,d in g.degree() if d == 1):
                noeuds = fct.apex2d3(apex,g)
                L = self.path_length(noeuds)
                if L < Reseau.SEUIL_DEPART*self.diameter:
                    if np.all([n not in g0.nodes for n in noeuds[1:]]):
                        for n in noeuds[1:]:#On ne change pas le d3
                            self.n2t[n] += 1
            nx.set_node_attributes(self.g, values = self.n2t, name = "t")
        return self
        
    def better_estimation_apex(self,npts:int=51,show:bool=False):
        """
        Return the reseau with a better estimation of the position of 
        the apexes.
        Change the position of apexes along the trajectories of branches to
        make them closer to the real change in time.
        """
        timeArray = np.asarray(Image.open(self.output_dir+f"Binarization/{self.name}_time_unscaled.tif"))
        corr= {}
        for b in self.branches:
            b.n2x = self.n2x
            b.n2y = self.n2y
            b.n2t = self.n2t
            n2index = {n:i for i,n in enumerate(b.noeuds)}
            length = b.s
            for t,a in zip(*b.get_all_apex()):
                ia = n2index[a]
                if a<0 or ia != len(length)-1:
                    n1 = b.noeuds[ia-1]
                    n2 = b.noeuds[ia+1]
                    pos1 = np.array([self.n2x[n1],self.n2y[n1]])
                    posA = np.array([self.n2x[a],self.n2y[a]])
                    pos2 = np.array([self.n2x[n2],self.n2y[n2]])
                    s = np.linspace(0,1,npts)
                    curve_x = pos1[0]*(s-1)*(s-1)-2*s*(s-1)*posA[0]+s*s*pos2[0]
                    curve_y = pos1[1]*(s-1)*(s-1)-2*s*(s-1)*posA[1]+s*s*pos2[1]
                    curve_t = np.array([timeArray[int(y),int(x)] for x,y in zip(curve_x,curve_y)])
                    if np.any(curve_t<= t) and np.any(curve_t>t):
                        i_ = next((i for i,t_ in enumerate(curve_t[:-1]) if curve_t[i]>t),-1)#next(i for i,t_ in enumerate(curve_t[:-1]) if curve_t[i+1]>t)#plutôt prendre le dernier de la liste qui est au bon temps
                        if i_ < 0:
                            continue
                        new_posA = np.array([curve_x[i_],curve_y[i_]])
                        corr[a] = [(self.n2x[a],self.n2y[a],self.n2t[a]),
                                   (new_posA[0],new_posA[1],t)]
                        self.n2x[a] = new_posA[0]
                        self.n2y[a] = new_posA[1]
                        self.n2t[a] = t
                        delta = np.sqrt(np.dot(new_posA-posA,new_posA-posA))
                        if show and delta>self.diameter:
                            fig,ax = plt.subplots(figsize=(3,3),layout="tight")
                            ax.imshow(self.image_at(t),cmap="Greys")
                            ax.plot(curve_x,curve_y)
                            ax.scatter([pos1[0],posA[0],pos2[0]],
                                    [pos1[1],posA[1],pos2[1]],
                                    marker="x",color=["red","gold","red"])
                            ax.scatter(curve_x[i_],curve_y[i_],marker="s",color="green")
                            ax.set_xlim(np.min(curve_x)-20,np.max(curve_x)+20)
                            ax.set_ylim(np.min(curve_y)-20,np.max(curve_y)+20)
                            ax.axis('off')
                            fig.savefig(self.output_dir+f"Branches/adjust_apex_{a}.png",dpi = 300)
                            plt.close()
            b.n2x = self.n2x 
            b.n2y = self.n2y 
            b.n2t = self.n2t
        self.correctif_apex = corr
        return self
    """
    ===========================================================================
    Dynamique
    """
    def dynamique(self):
        """
        Calcule la coordonnée temporelle pour chaque noeud du réseau.
        """
        print()
        print("-"*80)
        print("\t\tTemporal coordinates estimation...")
        dX = self.n2x
        dY = self.n2y
        dPos = {n: np.array([dX[n],dY[n]]) for n in self.g.nodes}
        print("\tEstimation of pixel time coordinate for last binarized image.")
        Nx,Ny = self.last.shape
        Nf = self.end - self.start + 1
        activated = np.where(self.last.reshape(Nx*Ny) > 0)

        tps = np.zeros(Nx*Ny,dtype=np.uint8)
        lum = np.transpose(np.array([np.asarray(Image.open(self.imgs.format(f))).flatten()
                                     for f in tqdm(range(self.start,self.end+1))],
                                     dtype=np.uint8))
        subsets = np.array_split(activated[0],20)                               #On découpe le problème en X subsets
        for pixels in tqdm(subsets,                                             #Ici X=20
                           desc="Promenade parmi les pixels"):                  #
            Ib = np.array([np.mean(lum[pixels,:T], axis = 1)
                           for T in range(1,Nf)])
            Ih = np.array([np.mean(lum[pixels,T:], axis = 1)
                           for T in range(1,Nf)])
            G = np.array([np.var(lum[pixels,:T], axis = 1)+
                          np.var(lum[pixels,T:], axis = 1)
                          for T in range(1,Nf)])
            G = G + np.where(Ih<Ib,255*Nf,0)
            Gnew = np.transpose(G)
            tps[pixels] = np.argmin(Gnew, axis = 1)+ self.start+1
        tps = np.reshape(tps,(Nx,Ny))
        #On fixe également les points initiaux suivant self.first
        init = np.where(self.first>0)
        tps[init] = self.start
        print("\tSaving this estimation as a tif image in Binarization folder.")
        scaledTps = int(255/Nf)*np.where(self.last.reshape(Nx,Ny)>0,
                                          tps-self.start+1,
                                          0)
        Image.fromarray(scaledTps).save(self.output_dir+f"Binarization/{self.name}_time_scaled.tif")
        Image.fromarray(tps).save(self.output_dir+f"Binarization/{self.name}_time_unscaled.tif")
        print("\tInitial estimation of node time coordinate.")
        self.n2t = {n: fct.node2time(self.g,n,tps,dPos,r=self.diameter//2) 
                    for n in tqdm(self.g.nodes)}
        """
        print("\tCorrection préalable des noeuds de degré 3")
        #Cette étape avait du sens au moment de sa création                     A couper ?
        #Maintenant je n'en suis plus si sûr...                                  v
        d3 = (n for n,d in self.g.degree if d == 3)                             #|
        for n in d3:                                                            #|
            t = int(self.n2t[n])                                                #|
            voisins = self.g.neighbors(n)                                       #|
            t1,t2,t3 = np.sort([int(self.n2t[v]) for v in voisins])             #|
            m = int((t1+t2)/2)                                                  #|
            self.n2t[n] = min(m,t)                                              #|
        #S'il faut couper c'est jusqu'ici.                                       ^
        """
        nx.set_node_attributes(self.g,values=self.n2t,name="t")
        print("\tExtracting starting network")
        start = self.startingGraph()
        bActifs = {n : False for n in self.g.nodes}
        for n in start.nodes:
            bActifs[n] = True

        print("\tIterative network reconstruction...")
        time = np.arange(self.start,self.end+1)
        nboucles = np.zeros_like(time)
        nnoeuds = np.zeros_like(time)
        erreurs = []
        for k,f in tqdm(enumerate(range(self.start,self.end+1))):
            again = True
            b = 0
            before = bActifs.copy()
            while(again):
                bActifsUp = fct.updateActifs(bActifs,self.g,self.n2t,f)
                again = np.any([bActifsUp[n]!=bActifs[n] for n in self.g.nodes])
                bActifs = bActifsUp.copy()
                b += 1
            nboucles[k] = b
            #print(f"\nframe{f:2d}, {b:3d} boucles")
            after = bActifs.copy()
            new = [n for n in self.g.nodes if after[n] and not before[n]]
            for n in new:
                self.n2t[n] = f
            #print(f"\t{len(new)} nouveaux noeuds actifs")       
            nnoeuds[k] = len(new)               
            if b >= Reseau.SEUIL_BOUCLE_DYNAMIQUE: #Si grd nb de boucles, c'est louche...
                #Enregistrement des erreurs potentielles
                g_new = self.g.subgraph(new)
                for k,amas in enumerate((g_new.subgraph(c).copy() 
                                         for c in nx.connected_components(g_new)
                                         if len(c) >= Reseau.SEUIL_BOUCLE_DYNAMIQUE)):
                    fig, ax = plt.subplots(figsize=(12,12))
                    fct.show(amas,ax)
                    xlim = plt.xlim()
                    ylim = plt.ylim()
                    ax.imshow(np.asarray(Image.open(self.imgs.format(f))),
                              cmap="Greys")
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    fig.suptitle(f"f = {f}, xlim = {xlim}, ylim = {ylim}")
                    fig.savefig(self.output_dir+"PotentialErrors/"+f"frame_{f}_n_{k}_"+
                                f"X_{xlim[0]:.0f}_Y_{ylim[0]:.0f}_"+
                                f"W_{xlim[1]-xlim[0]:.0f}_H_{ylim[1]-ylim[0]:.0f}"+".png")
                    plt.close(fig)
                    erreurs.append([f,
                                    int(xlim[0]),int(ylim[0]),
                                    int(xlim[1]-xlim[0]),int(ylim[1]-ylim[0])])     
            gframe = self.g.subgraph([n for n in self.g.nodes if bActifs[n]])   
            nx.set_node_attributes(self.g,self.n2t,"t")
            fig,ax  = plt.subplots(figsize=(8,8))
            fct.show(gframe,ax)
            fig.savefig(self.output_dir+f"GraphesVisu/graph_{f:02d}.png",dpi=300)
            plt.close(fig)
        erreurs = np.array(erreurs,dtype=int)
        np.savetxt(self.output_dir+"PotentialErrors/"+
                   "list_errors_positions.txt",
                   erreurs,
                   fmt="%d",
                   delimiter="\t",
                   header="F\tX\tY\tW\tH")
        fig,ax1 = plt.subplots(figsize=(3.1,2.8))
        color = "red"
        ax1.set_xlabel("$t\quad[T]$")
        ax1.set_ylabel("$N_{iteration}$", color = color)
        ax1.plot(time,nboucles,color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.hlines(Reseau.SEUIL_BOUCLE_DYNAMIQUE,
                   xmin=time[0],xmax=time[-1],
                   colors=color,linestyles='dashed')
        ax2 = ax1.twinx()
        color='blue'
        ax2.set_ylabel("$N_{nodes}(t)-N_{nodes}(t-T)$", color = color)
        f = np.where(nnoeuds>0)
        ax2.plot(time[f],nnoeuds[f],color=color)
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor=color)
        fig.tight_layout()
        fig.savefig(self.output_dir+"datation.png",dpi=300)
        plt.close(fig)
        np.savetxt(self.output_dir+"datation.txt",
                   np.array([time,nboucles,nnoeuds]).T,
                   header="#t\tNit\tNnodes",delimiter="\t")
        #Correction des derniers noeuds restant comme étant à la dernière frame
        for n in self.g.nodes:
            if not bActifs[n]:
                self.n2t[n] = self.end
        nx.set_node_attributes(self.g,self.n2t,"t")
        print("\t\tEnd of time estimations. Check folder 'PotentialErrors/' if necessary.")
        print(f"{'-':-^80}")
        print()
        return self
    
    def startingGraph(self):
        """
        Renvoie le graph initial "start"
        """
        toKeep = [n for n in self.g.nodes if self.n2t[n]<=self.start]
        start = self.g.subgraph(toKeep).copy()
        #Si le réseau est deconnecté, on ne prend que le plus gros morceau...
        start = start.subgraph(sorted(nx.connected_components(start),
                               key=len,reverse=True)[0]).copy()
        return start
    
    """
    ===========================================================================
    Orientation
    """
    def orient_and_branches(self,
                            CLUSTER_SIZE_MAX:int = 10,
                            r_diameter_unit:int = 3,
                            alpha:float=1.,
                            show:bool = False,
                            orient_method:str = "MC method",
                            try_reverse:bool=False):
        """
        Create twigs then orient them before it identify the branches.
        CLUSTER_SIZE_MAX = Maximum size for twig cluster to orient by trying 
                           all configurations
        r_diameter_unit = Radius in unit unit of hyphal diameter to determine 
                          the orientation vectors
        show = switch between showing the different step in orienting the twigs
        Has some function inside that don't have vocation to be use elsewhere.
        """
        #Initialisation:
        dX, dY, dT = (nx.get_node_attributes(self.g,coord)
                      for coord in ("x","y","t"))
        r = r_diameter_unit*self.diameter
        #Create twigs
        def create_twigs()->int:
            """
            Create twigs. Overwrite pre-existing twigs.
            Return the number of twigs created.
            """
            d2 = [n for n,d in self.g.degree if d == 2]
            i = 0
            self.brindilles = []#Initialise twigs as empty sequence
            for u,v in tqdm(self.g.edges,
                            desc = "Search for d3-d3(1) twigs... "):
                du = self.g.degree(u)
                dv = self.g.degree(v)
                if ((du == 3 or du == 1) and (dv == 3 or dv == 1)):
                    self.brindilles.append(Brindille(index = i,
                                                    noeuds = [u,v],
                                                    n2x = self.n2x,
                                                    n2y = self.n2y,
                                                    n2t = self.n2t,
                                                    inBranches = []))
                    i += 1
            print(f"{i} twig(s) made of exactly 2 degree 3 (or 1) nodes.")
            toMatch = {n:True for n in d2}
            for n in tqdm((n for n in d2 if toMatch[n]),
                        desc="Identification of twigs for each degree 2 node ... "):
                brin = fct.find_twig(n,self.g)
                if brin[0] == brin[-1]:
                    self.g.remove_nodes_from(brin[1:-1])
                    print(f"I removed {brin[1:-1]} because twig with start point = end point")
                    for u in brin[1:-1]:
                        toMatch[u] = False
                    continue
                self.brindilles.append(Brindille(index = i,
                                                noeuds = brin,
                                                n2x = self.n2x,
                                                n2y = self.n2y,
                                                n2t = self.n2t,
                                                inBranches = []))
                i += 1
                for k in brin:
                    toMatch[k] = False
            print(f"{i} twigs created.")
            return i
        create_twigs()
        #initialisation of twigs status
        toOrient = {b.index: True for b in self.brindilles}
        dbrindilles = {n:[] for n,d in self.g.degree if d != 2}
        for b in tqdm(self.brindilles,
                      desc="Initialisation of dictionnary n->[twigs] ... "):
            dbrindilles[b.noeuds[ 0]].append(b)
            dbrindilles[b.noeuds[-1]].append(b)
        n2brindilles = {n: [] for n in self.g.nodes}
        for b in self.brindilles:
            for n in b.noeuds:
                n2brindilles[n].append(b)
        
        def show_current_status(title:str=""):
            """
            Show the current status in orienting the different twigs.
            """
            fig,ax = plt.subplots(figsize=(8,8))
            X = []
            Y = []
            U = []
            V = []
            C = []
            for b in self.brindilles:
                X.append(dX[b.noeuds[0]])
                Y.append(dY[b.noeuds[0]])
                U.append(dX[b.noeuds[-1]]-dX[b.noeuds[0]])
                V.append(dY[b.noeuds[-1]]-dY[b.noeuds[0]])
                C.append("red" if toOrient[b.index] else "green")
            ax.quiver(X, Y, U, V, color = C,
                       angles = 'xy', scale_units = 'xy', scale = 1)
            ax.set_title(title)
            ax.set_axis_off()
            plt.show()
            plt.close(fig)
        
        if show:
            show_current_status("Start of orientation")
        #Identify the source
        source = self.source if self.source is not None else self.find_source()
        #Fixed orientation (source and d1 rule)
        def set_fixed_orientation()->int:
            """
            Fixes the orientation of the source's twigs and terminal twigs 
            (ending by d1) and propagate this information to avoid any other
            source.
            Return the number of twigs fixed.
            """
            i = 0
            #Source's twigs
            for b in n2brindilles[source]:
                if b.noeuds[-1] == source:
                    b.noeuds.reverse()
                b.confiance = 1
                toOrient[b.index] = False
                i += 1
            #Apex
            for apex in (n for n,d in self.g.degree if d == 1):
                b = n2brindilles[apex][0]
                if b.noeuds[0] == apex:
                    b.noeuds.reverse()
                b.confiance = 1.
                toOrient[b.index] = False
                i += 1
            #Propagation
            propagation = True
            while (propagation):
                propagation = False 
                for n in (n for n,d in self.g.degree 
                        if (d == 3 and sum(toOrient[b.index] 
                                            for b in n2brindilles[n])==1)):
                    #On a un degré 3 avec uniquement 1 brindille à orienté.
                    nsortie = sum(b.noeuds[0] == n 
                                for b in n2brindilles[n]
                                if not toOrient[b.index])
                    if nsortie == 2:
                        #La dernière brindille doit être une entrée.
                        bi = next(b for b in n2brindilles[n] 
                                if toOrient[b.index])
                        toOrient[bi.index] = False
                        if bi.noeuds[0] == n:
                            bi.noeuds.reverse()
                        bi.confiance = 1.
                        i += 1
                        propagation = True
            return i
        set_fixed_orientation()
        if show:
            show_current_status("Fixed twigs")
        #Orientation by growth chain
        def orient_growth_chains()->int:
            """
            Orientation according to growth without ambiguities.
            Return the number of twigs oriented this way.
            """
            chaines = self.chaines_croissance()
            i = 0
            for chaine in tqdm(chaines,
                            desc="Alignement selon les chaînes de croissance :"):
                bs = set([b for n in chaine for b in n2brindilles[n]])
                for b in (b for b in bs if toOrient[b.index]):
                    intersection1 = np.array([n for n in b.noeuds 
                                            if n in chaine])
                    if len(intersection1)>1:
                        intersection2 = np.array([n for n in chaine 
                                                if n in b.noeuds])
                        if np.any(intersection1 != intersection2):
                            #Permet de comparer le sens de parcourt de la brindille
                            #par rapport à celui de la chaine
                            b.reverse()
                        toOrient[b.index] = False
                        b.calcul_confiance(seuil=Reseau.SEUIL_LAT)
                        i += 1
            return i
        orient_growth_chains()
        if show:
            show_current_status("Growth direction")
        #Remaining orientation (all configuration or MC_method if needed)
        #Identification of remaining twigs
        i2b = {b.index:b for b in self.brindilles}
        gbrindilles = nx.Graph()
        gbrindilles.add_nodes_from((b.index for b in self.brindilles 
                                    if toOrient[b.index]))
        for b1 in (b for b in self.brindilles if toOrient[b.index]):
            for b2 in (b for n_ext in (b1.noeuds[0],b1.noeuds[-1])
                        for b in n2brindilles[n_ext]
                        if b.index != b1.index and toOrient[b.index]):
                gbrindilles.add_edge(b1.index,b2.index)
        #The size of the biggest cluster determine which method we use.
        clusters_size = [len(cluster) 
                         for cluster in nx.connected_components(gbrindilles)]
        largest_size = np.max(clusters_size)
        def test_all_config()->int:
            """
            Orient all twigs that remain by trying all configuration and 
            choosing the one which maximize the score (alignment and time)
            Return the number of oriented twigs.
            """
            i = 0
            #First, let's consider isolated twig
            def alignement(bi)->float:
                """
                Calcule un score d'alignement d'une brindille avec ses voisines
                """
                score_alignement = 0.
                nstart,nend = (bi.noeuds[ext] for ext in (0,-1))
                for n_ext in (nstart,nend):
                    ui = bi.unit_vector(n_ext,r,dX,dY)
                    voisins = (b for b in n2brindilles[n_ext] if b.index!=bi.index)
                    for bj in voisins:
                        uj = bj.unit_vector(n_ext,r,dX,dY)
                        score_alignement += np.dot(ui,uj)
                return score_alignement
            singletons = [i2b[index] for index,d in gbrindilles.degree 
                          if d == 0]
            for singleton in tqdm(singletons,
                                desc="Alignment of isolated twigs ..."):
                alignement_sens1 = alignement(singleton)
                if alignement_sens1<0:
                    singleton.reverse()
                toOrient[singleton.index] = False
                i += 1
            gbrindilles.remove_nodes_from((b.index for b in singletons))
            #Calculation of confidences
            for b in tqdm(self.brindilles,desc="Calcul des confiances : "):
                if toOrient[b.index]:
                    b.calcul_confiance(dT)
                else:
                    b.confiance = 1.
            #Now let's fix what remains.
            clusters = nx.connected_components(gbrindilles)
            def score_cluster(cluster, alpha=1.)->float:
                """
                Calcule un score combinant confiance et alignement pour un cluster
                d'intérêt.
                Le dosage entre les deux est réglé par le paramètre alpha.
                """
                score = 0
                for bi in cluster:
                    score += bi.confiance
                    alignement = sum(np.dot(bi.unit_vector(n_ext,r,dX,dY),
                                            bj.unit_vector(n_ext,r,dX,dY))*bj.confiance 
                                    for n_ext in (bi.noeuds[0],bi.noeuds[-1])
                                    for bj in n2brindilles[n_ext]
                                    if bi.index != bj.index)
                    score += alpha*(1-np.abs(bi.confiance))*alignement
                return score

            def traite_cluster(cluster):
                size = len(cluster)
                score_max = 0
                config_max = f"{0:0{size}b}"
                for k in range(2**size):
                    config = f"{k:0{size}b}"
                    #Chargement de la configuration
                    for i,bi in zip(config,brindilles):
                        if i == '1':
                            bi.reverse()
                    #Calcul du score de la configuration actuelle
                    score_config = score_cluster(brindilles)
                    if score_config > score_max or k == 0:
                        score_max = score_config
                        config_max = config
                    #Remise à zero de la configuration
                    for i,bi in zip(config,brindilles):
                        if i == '1':
                            bi.reverse()
                #Chargement de la configuration au score maximal
                for i,bi in zip(config_max,brindilles):
                    toOrient[bi.index] = False
                    if i == '1':
                        bi.reverse()
                return 1

            for cluster in tqdm(clusters,
                                desc="Alignement des clusters : "):
                #On explore toutes les configurations et on prend celle
                #au plus grand score.
                brindilles = [i2b[i] for i in cluster]
                traite_cluster(brindilles)
                i += len(cluster)
            return i
        
        def MC_exploration(alpha = alpha)->int:
            """
            Orient all twigs that remain by maximising the score (alignement 
            and time) using a MonteCarlo method to explore the possibilities
            alpha : ratio between time correlation and spacial alignment.
            """
            orientables = [b for b in self.brindilles if toOrient[b.index]]
            nOrientables = len(orientables)
            #We want to exclude the case where all twigs are already oriented
            if nOrientables:
                #Calculation of confidences
                for b in tqdm((b for b in self.brindilles 
                               if toOrient[b.index]),
                              desc= "Calculation of twigs' confidence ... "):
                    b.calcul_confiance(seuil=Reseau.SEUIL_LAT)
                #Calculation of vectors
                n2b2vect = {n:{} for n in self.g.nodes}
                unitvect = {}
                for b in tqdm(self.brindilles,
                            desc = "Calculation of twigs' vectors ... "):
                    u0 = b.unit_vector(b.noeuds[+0],r=r)
                    u1 = b.unit_vector(b.noeuds[-1],r=r)
                    n2b2vect[b.noeuds[+0]][b.index] = u0
                    n2b2vect[b.noeuds[-1]][b.index] = u1
                    debut, fin = b.noeuds[0], b.noeuds[-1]
                    v = np.array([self.n2x[fin]-self.n2x[debut],
                                  self.n2y[fin]-self.n2y[debut]])
                    unitvect[b.index] = v / np.linalg.norm(v)
                
                #Definition of a score for each twig
                def score(bi, a = alpha): 
                    ci = bi.confiance
                    n0,n1 = bi.noeuds[0],bi.noeuds[-1]
                    align = 0
                    compteur_voisins = 0
                    for n_ext in (n0,n1):
                        for bj in (bj for bj in n2brindilles[n_ext]
                                if bj.index != bi.index):
                            align += np.dot(n2b2vect[n_ext][bi.index],
                                            n2b2vect[n_ext][bj.index])*bj.confiance
                            compteur_voisins += 1
                    align = align#/compteur_voisins
                    return ci + a*(1-np.abs(ci))*align
                score_actuel = sum(score(b) for b in self.brindilles)
                iterations = 100
                betamin = 1
                betamax = 1024
                beta = betamin
                scores = []
                print(f"Score initial = {score_actuel:.2f}") 
                score_max = score_actuel
                #On cherche à maximiser ce score -> simulation Monte Carlo.
                config_max = {b.index:(b.noeuds[0],b.noeuds[-1])
                            for b in self.brindilles}
                config_init = {b.index:(b.noeuds[0],b.noeuds[-1])
                            for b in self.brindilles}
                #Garde une trace de l'orientation de chaque brindille dans
                #le cas ayant le score maximal

                acc_moyenne = 100
                
                compteur = 0
                while(acc_moyenne>1. and beta <= betamax):
                    acc_moyenne = 0
                    for i in range(iterations):
                        acceptance = 0
                        order = np.arange(nOrientables)
                        np.random.shuffle(order)
                        for k_change in order:
                            bi = orientables[k_change]
                            delta_si = -2*score(bi)
                            if np.random.random() < np.exp(delta_si*beta):
                                #On accepte le changement
                                n2b2vect[bi.noeuds[0]][bi.index] = -n2b2vect[bi.noeuds[0]][bi.index]
                                n2b2vect[bi.noeuds[-1]][bi.index] = -n2b2vect[bi.noeuds[-1]][bi.index]
                                unitvect[bi.index] = -unitvect[bi.index]
                                bi.confiance = -bi.confiance
                                bi.noeuds.reverse()
                                score_actuel += delta_si
                                acceptance += 1
                        if score_actuel > score_max:
                            #On change la config 
                            config_max = {b.index:(b.noeuds[0],b.noeuds[-1])
                                        for b in self.brindilles}
                            score_max = score_actuel
                        acceptance = acceptance/nOrientables*100
                        acc_moyenne += acceptance
                        scores.append(score_actuel) 
                    acc_moyenne = acc_moyenne/iterations
                    print((f"Etape beta = {beta:.2f} : "
                        f"acceptance moyenne = {acc_moyenne:.2f}%"))
                    beta = beta*2
                    compteur += 1
                #Une fois tout fini, je doisrecharger la configuration 
                # au score maximal
                for b in self.brindilles:
                    ndebut,nfin = config_max[b.index]
                    if ndebut == b.noeuds[-1] or nfin == b.noeuds[0]:
                        b.noeuds.reverse()
                        n2b2vect[ndebut][b.index] = -n2b2vect[ndebut][b.index] 
                        n2b2vect[nfin][b.index] = -n2b2vect[nfin][b.index]
                
                #A voir si c'est utile
                #Trace l'évolution du score total du réseau en fonction 
                #des itérations successives.
                fig = plt.figure(f"Evolution du score lors de l'alignement.",
                                figsize=(12,8))
                ax = fig.add_subplot()
                ax.plot(scores)
                ax.vlines([i*iterations for i in range(compteur)],
                        ymin = np.min(scores), ymax = np.max(scores),
                        linestyles="dashed",colors="black")
                for i in range(compteur):
                    subscores= scores[i*iterations:(i+1)*iterations]
                    ax.text((i+1/2)*iterations,
                            (np.mean(subscores)+2*np.std(subscores)),
                            f"$\\beta =$ {2**i}",
                            fontdict={"ha":"center"})
                ax.set_xlabel("cycle d'itérations")
                ax.set_ylabel("Score")
                fig.suptitle(("Evolution du score lors de l'alignement\n"
                            f"Score maximal : {score_max:.2f}"))
                fig.savefig(self.output_dir+f"Score_alignement.pdf")
                plt.close(fig)
            
            #A voir si c'est utile
            #Place dans l'espace (Alignement, confiance) chaque brindille.
            confiances = np.zeros(len(self.brindilles))
            alignements = np.zeros(len(self.brindilles))
            for k, brindille in enumerate(self.brindilles):
                confiances[k] = brindille.confiance
                align = 0
                for nextremite in (brindille.noeuds[0],brindille.noeuds[-1]):
                    for b_voisine in (b for b in n2brindilles[nextremite] 
                                    if b.index != brindille.index):
                        align += np.dot(n2b2vect[nextremite][brindille.index],
                                        n2b2vect[nextremite][b_voisine.index])*b_voisine.confiance
                alignements[k] = align
            
            fig,ax = plt.subplots(figsize=(11.69,8.27))
            ax.scatter(alignements,confiances)
            ax.set_ylim([-1.1,+1.1])
            ax.set_xlim([-4.4,+4.4])
            ax.set_xlabel("alignement")
            ax.set_ylabel("confiance")
            fig.savefig(self.output_dir+"distribution_confiance_alignement.jpg",
                        dpi=300)
            #plt.show()
            plt.close()

            return 1
        
    
        if orient_method == "All configuration":
            if largest_size <= CLUSTER_SIZE_MAX:
                print("Trying to orient with 'All configuration'.")
                print("If it doesn't work try 'MC method'.")
                test_all_config()
            else:
                print("Too large for 'All configuration' method.")
                print("Trying to orient with 'MC method'.")
                print("If it doesn't work try 'All configuration'.")
                MC_exploration()
        elif orient_method == "MC method":
            print("Trying to orient with 'MC method'.")
            print("If it doesn't work try 'All configuration'.")
            MC_exploration()

        if show:
            show_current_status("Remaining orientation")
        #Resolve potential issues - Security to implement to avoid recursive problems
        def solve_extra_sources()->int:
            """
            Only one source should remains at the end.
            Reorient (recursively) the least confident twig at each 
            extra source.
            Return the number of sources corrected this way.
            """
            sources = []
            puits = []
            for n in (n for n,d in self.g.degree if d == 3 and n != source):
                n_entrees = sum(b.noeuds[-1] == n for b in n2brindilles[n])
                if n_entrees == 0:
                    sources.append(n)
                if n_entrees == 3:
                    puits.append(n)
            print(f"There is {len(sources)} extra source(s).")
            print(f"Also there is currently {len(puits)} well.")
            def score(bi, a = 1.): 
                    ci = bi.confiance
                    n0,n1 = bi.noeuds[0],bi.noeuds[-1]
                    align = 0
                    compteur_voisins = 0
                    for n_ext in (n0,n1):
                        for bj in (bj for bj in n2brindilles[n_ext]
                                if bj.index != bi.index):
                            align += np.dot(bi.unit_vector(n_ext,r),
                                            bj.unit_vector(n_ext,r))*bj.confiance
                            compteur_voisins += 1
                    align = align#/compteur_voisins
                    return ci + a*(1-np.abs(ci))*align
            possible = {b.index: b.confiance<1. for b in self.brindilles}
            def recursive_source(n,brindilles2change = [],dscore = 0):
                n_entrees = sum(b.noeuds[-1] == n for b in n2brindilles[n])
                if n_entrees != 0:#Si on n'a plus une source en n
                    return brindilles2change,dscore
                options = [b for b in n2brindilles[n]
                           if (possible[b.index] and
                               b.index not in brindilles2change)]
                if options:
                    solutions = []
                    for b in options:
                        nb = b.noeuds[-1]
                        sb = -2*score(b)
                        listeb = [*brindilles2change, b.index]
                        solution_b,score_b = recursive_source(nb,
                                                            brindilles2change=listeb,
                                                            dscore = dscore+sb)
                        solutions.append((solution_b,score_b))
                    solutions = sorted(solutions,key = lambda tup: tup[1],
                                    reverse=True)
                    return solutions[0]
                dscore += -1000
                return brindilles2change, dscore
            compteur = 0
            while (sources):
                compteur += 1
                n = sources[0]
                solution,sc = recursive_source(n)
                for bidx in solution:
                    b = i2b[bidx]
                    b.noeuds.reverse()
                    b.confiance = - b.confiance
                    toOrient[b.index] = False
                    possible[b.index] = False
                sources = [n for n,d in self.g.degree 
                           if (d == 3 and
                               n != source and
                               sum(b.noeuds[-1]==n for b in n2brindilles[n]) == 0)]
            print(f"We have solved {compteur} source(s).")
            return compteur
        solve_extra_sources()
        if show:
            show_current_status("Issue(s) solved")
        #Create branches
        def start_branching()->int:
            """
            Create the different branches.
            Return the number of branches created.
            """
            self.branches = []
            n2brindilles = {n:[] for n in self.g.nodes}
            for b in self.brindilles:
                for n in b.noeuds:
                    n2brindilles[n].append(b)
            """
            ========================================================================
            ====================  Appariement des brindilles  ======================
            ========================================================================
            """
            paires = {n: {} for n in (n for n,d in self.g.degree if d >= 3)}
            for n in tqdm((n for n,d in self.g.degree if d >= 3 and n != source),
                        desc="Appariement des entrées/sorties ... "):
                es = [b for b in n2brindilles[n] if b.noeuds[-1] == n]
                ss = [b for b in n2brindilles[n] if b.noeuds[ 0] == n]
                if len(es) == 0:#Si problème on le visualise
                    print(f"Problème en ({dX[n]},{dY[n]})")
                    print(es,ss)
                    noi = set([*[b.noeuds[ 0] for b in [*es,*ss]],
                               *[b.noeuds[-1] for b in [*es,*ss]]])
                    twigsoi = []
                    for u in noi:
                        twigsoi = [*twigsoi,*n2brindilles[u]]
                    twigsoi = set(twigsoi)
                    fig,ax = plt.subplots()
                    self.show_brindilles_light(ax = ax,
                                            condition = lambda b: b in n2brindilles[n],
                                            title = f"Erreur en {n}",
                                            only= twigsoi)
                    fig.savefig(self.output_dir+f"ErreurOrientation_{n}.png")
                    plt.close(fig)
                paires[n] = fct.appariement(self,es,ss,r = r,
                                            seuil_lat = Reseau.SEUIL_LAT)
            if try_reverse:
                compteur_reverse = 0
                for twig in (twig for twig in self.brindilles if twig.confiance<=0):
                    n1,n2 = twig.noeuds[0],twig.noeuds[-1]
                    if self.g.degree(n2) == 1:
                        continue
                    if (paires[n2][twig.index] == "stop") and (twig in paires[n1]["newApi"] or twig in paires[n1]["newLat"]):
                        try:
                            twig.reverse()
                            compteur_reverse += 1
                            for n in (n1,n2):
                                es = [b for b in n2brindilles[n] if b.noeuds[-1] == n]
                                ss = [b for b in n2brindilles[n] if b.noeuds[ 0] == n]
                                paires[n] = fct.appariement(self,es,ss,r=r,seuil_lat= Reseau.SEUIL_LAT)
                        except Exception:
                            twig.reverse()
                            compteur_reverse -= 1 
                            for n in (n1,n2):
                                es = [b for b in n2brindilles[n] if b.noeuds[-1] == n]
                                ss = [b for b in n2brindilles[n] if b.noeuds[ 0] == n]
                                paires[n] = fct.appariement(self,es,ss,r=r,seuil_lat= Reseau.SEUIL_LAT)

                print(f"Après l'appariement j'ai retourné {compteur_reverse} brindilles qui serait devenue des branches d'une brindille")
            
            branches = [Branche(index = k,
                                noeuds = s.noeuds,
                                n2x = self.n2x,
                                n2y = self.n2y,
                                n2t = self.n2t,
                                brindilles = [s.index],
                                nature = "Initial",
                                ending = "Growing")
                        for k,s in enumerate(n2brindilles[source])]
            index = 3
            for n in tqdm((n for n,d in self.g.degree if d == 3 and n != source),
                          desc="Création des branches ... "):
                croisement = paires[n]
                for brin in croisement["newApi"]:
                    branches.append(Branche(index = index,
                                            noeuds = brin.noeuds,
                                            n2x = self.n2x,
                                            n2y = self.n2y,
                                            n2t = self.n2t,
                                            brindilles = [brin.index],
                                            nature="Apical",
                                            ending="Growing"))
                    index += 1
                for brin in croisement["newLat"]:
                    branches.append(Branche(index = index,
                                        noeuds = brin.noeuds,
                                        n2x = self.n2x,
                                        n2y = self.n2y,
                                        n2t = self.n2t,
                                        brindilles = [brin.index],
                                        nature="Lateral",
                                        ending="Growing"))
                    index += 1
            compteur = 0
            enCroissance = branches
            while(enCroissance):
                Growing = []
                for b in tqdm(enCroissance,
                            desc=f"etape {compteur}, croissance ... "):
                    n = b.noeuds[-1]
                    if self.g.degree(n) == 1: #Ending by free apex
                        b.ending = "d1"
                        continue
                    croisement = paires[n]
                    if croisement[b.brindilles[-1]] == "stop":
                        #Ending by fusion
                        b.ending="Fusion?"
                        continue
                    else:
                        suite = croisement[b.brindilles[-1]]
                        b.grow(suite)
                        Growing.append(b)
                enCroissance = Growing
                compteur += 1
            self.branches = branches
            return len(branches)
        start_branching()
        return self
    
    def find_source(self)->int:
        """
        Identifie la source du réseau comme étant le barycentre du graphe initial
        où l'on ne considère pas les noeuds de degrés 2.
        Si ce n'est pas possible automatiquement, une selection manuelle est 
        alors proposée.
        """
        g0 = self.network_at(self.start)
        try:
            source = nx.barycenter(fct.prune(g0))[0]
            if g0.degree(source) <= 2:
                raise ValueError("Source is not a degree 3 node.")
        except ValueError:
            print("La source n'a pas pu être définie automatiquement.")
            print("Merci de choisir manuellement la source :")
            fig,ax = plt.subplots(figsize=(8,8))
            labeldict = {n:n if g0.degree(n) in [1,3] else "" 
                         for n in g0.nodes}
            dX,dY = (nx.get_node_attributes(g0,coord) 
                     for coord in ("x","y"))
            pos = {n:(dX[n],dY[n]) for n in g0.nodes}
            nx.draw(g0,pos=pos,ax=ax,labels=labeldict,show_labels=True)
            xlim = plt.xlim()
            ylim = plt.ylim()
            ax.imshow(self.imgs[self.start],cmap="Greys")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            plt.show()
            source = int(input('La source choisie est : '))
            while g0.source(source) <= 2:
                 source = int(input('Merci de choisir un noeud de degree 3 : '))
            plt.close(fig)
        self.source = source
        return source
    
    """
    ===========================================================================
    Chevauchements
    """
    def detect_overlaps(self, vmean,
                       seuilL_diameterUnit = 5,
                       seuilT = SEUIL_LAT):
        """
        Détection de tous les chevauchements potentiels.
        """
        seuilL = seuilL_diameterUnit * self.diameter
        b_voisines = self.neighborhood()
        dX = self.n2x
        dY = self.n2y
        dT = self.n2t

        def filtre_direct(e,s)->bool:
            """
            Filtre une paire entrée sortie suivant si elle est potentiellent
            un chevauchement 'simple'.
            """
            if np.any([n in e.noeuds for n in s.noeuds]):
                return False
            abs_e = e.abscisse_curviligne()
            abs_s = s.abscisse_curviligne()
            tt_e = [dT[n] for n in e.noeuds]#e.t
            tt_s = [dT[n] for n in s.noeuds]#s.t
            te = tt_e[-2]
            ts = tt_s[+1]
            deltat = ts - te
            if deltat < 0:
                return False
            if deltat > seuilT:
                return False
            deltaL = np.sqrt((dX[s.noeuds[0]]-dX[e.noeuds[-1]])**2+
                             (dY[s.noeuds[0]]-dY[e.noeuds[-1]])**2)
            if deltaL > seuilL or deltaL>vmean*(deltat+1):
                return False 
            #Alignement :
            abs_e = e.abscisse_curviligne()
            abs_s = s.abscisse_curviligne()
            abs_e = abs_e - abs_e[-1]
            ke = next((len(abs_e)-k-1 
                       for k,l in enumerate(reversed(abs_e)) 
                       if l<-seuilL),
                      0)
            ks = next((k 
                       for k,l in enumerate(abs_s) 
                       if l>seuilL),
                      -1)
            vect_e = np.array([dX[e.noeuds[-1]]-dX[e.noeuds[ke]],
                               dY[e.noeuds[-1]]-dY[e.noeuds[ke]]])
            vect_e = vect_e/np.linalg.norm(vect_e)
            vect_s = np.array([dX[s.noeuds[ks]]-dX[s.noeuds[0]],
                               dY[s.noeuds[ks]]-dY[s.noeuds[0]]])
            vect_s = vect_s/np.linalg.norm(vect_s)
            return np.dot(vect_e,vect_s)>0.5

        #Il faudrait afficher ses résultats pour voir ce que ça donne...
        liste_chevauchements  = []
        for b in tqdm(self.branches):
            voisines = b_voisines[b.index]
            #On doit tester si le noeud initial est à la fois une entrees et une sortie  potentielles.
            nodes_input, nodes_output = [e.noeuds[-1] for e in voisines["in"]],[s.noeuds[0] for s in voisines["out"]]

            for e,s in ((e,s) 
                        for e in voisines["in"]
                        for s in voisines["out"]
                        if filtre_direct(e,s) and e.noeuds[-1]!=b.noeuds[0]):
                liste_chevauchements.append((b,e,s))

        print(f"{len(liste_chevauchements)} potential overlap(s) detected.")
        graph_chevauchement = nx.Graph()
        graph_chevauchement.add_nodes_from((k for k,c in enumerate(liste_chevauchements)))
        connexions = [(k1,k2) 
                      for k1,c1 in enumerate(liste_chevauchements) 
                      for k2,c2 in enumerate(liste_chevauchements)
                      if (k1 != k2 and
                          (c1[1].index == c2[1].index or
                           c1[2].index == c2[2].index))]
        graph_chevauchement.add_edges_from(connexions)
        liste_chevauchements_filtree = [liste_chevauchements[c] 
                                        for c,d in graph_chevauchement.degree() 
                                        if d == 0]
        for composant in (c for c in nx.connected_components(graph_chevauchement)
                          if len(c)>1):
            #composant = Set des index des chevauchements
            #Il faut départager les chevauchements parmi ceux du composant.
            #toOrder.append(some_function_to_find_best(composant)) 
            composant = list(composant)
            nchevauchement = len(composant)
            scores = np.zeros(nchevauchement)
            for i,k in enumerate(composant):
                chevauchement = liste_chevauchements[k] 
                bp,be,bs = chevauchement
                #chevauchement = (bp,be,bs)
                #On départage en fonction de :
                #L'alignement be.bs :
                abs_e = be.abscisse_curviligne()
                abs_e = abs_e - abs_e[-1]
                abs_s = bs.abscisse_curviligne()
                nstop_e = next((n
                                for n,s in zip(reversed(be.noeuds),
                                               reversed(abs_e))
                                if s < - seuilL),
                                be.noeuds[0])
                nstop_s = next((n 
                                for n,s in zip(bs.noeuds,abs_s)
                                if s > seuilL),
                                bs.noeuds[-1])
                u_e = np.array([dX[be.noeuds[-1]]-dX[nstop_e],
                                dY[be.noeuds[-1]]-dY[nstop_e]])
                u_s = np.array([dX[nstop_s]-dX[bs.noeuds[0]],
                                dY[nstop_s]-dY[bs.noeuds[0]]])
                u_e = u_e/np.linalg.norm(u_e)
                u_s = u_s/np.linalg.norm(u_s)
                scores[i] = np.dot(u_e,u_s)
            i_max = np.argmax(scores)
            chevauchement = liste_chevauchements[composant[i_max]]
            liste_chevauchements_filtree.append(chevauchement)
        def visionnage(b,e,s,show=False):
            """
            Affichage du chevauchement de la branche porteuse b par la 
            branche entrante e et la branche sortante s.
            """
            #Première étape, déterminer te et ts
            tt_e = [dT[n] for n in e.noeuds]
            tt_s = [dT[n] for n in s.noeuds]
            tt_b = [dT[n] for n in b.noeuds]
            xy_e = np.array([[dX[n],dY[n]] for n in e.noeuds])
            xy_s = np.array([[dX[n],dY[n]] for n in s.noeuds])
            xy_b = np.array([[dX[n],dY[n]] for n in b.noeuds])
            xy_centre = (xy_e[-1,:]+xy_s[0,:])*.5
            rayon_vision = 50
            #Détermination des instants du chevauchement
            te = tt_e[-2]
            ts = tt_s[+1]
            #Il faut qu'on affiche l'état du réseau aux frames allant
            #de max(te-1,start) à min(ts+1,end)
            frames = np.arange(max(te-1,self.start),min(ts+1,self.end)+1)
            #Affichage de l'image aux différentes frames d'intérêts.
            for f in frames:
                kstop_e,kstop_s,kstop_b = [next((k for k, t in enumerate(tt) 
                                                 if t > f),
                                                len(tt))
                                           for tt in (tt_e,tt_s,tt_b)]
                
            deltaL = np.sqrt(np.sum((xy_e[-1,:]-xy_s[0,:])**2))
            abs_e,abs_s = [branche.abscisse_curviligne(dX = dX, dY = dY)
                           for branche in (e,s)]
            abs_s = abs_s + abs_e[-1] + deltaL
            if show:
                fig,ax = plt.subplot_mosaic([[f"{f}" for f in frames],
                                            ["abs" for f in frames]],
                                            figsize=(11.69,8.27),
                                            constrained_layout=True)
                for f in frames:
                    ax[f"{f}"].plot(xy_e[:kstop_e,0],xy_e[:kstop_e,1],
                                color="chartreuse")
                    ax[f"{f}"].plot(xy_s[:kstop_s,0],xy_s[:kstop_s,1],
                                    color="plum")
                    ax[f"{f}"].plot(xy_b[:kstop_b,0],xy_b[:kstop_b,1],
                                    color="orange")             
                    ax[f"{f}"].imshow(self.imgs[f],cmap="Greys")
                    ax[f"{f}"].set_title(f"frame {f}")
                    ax[f"{f}"].set_xlim(xy_centre[0]-rayon_vision,
                                        xy_centre[0]+rayon_vision)
                    ax[f"{f}"].set_ylim(xy_centre[1]-rayon_vision,
                                        xy_centre[1]+rayon_vision)
                ax["abs"].vlines([abs_e[-1],abs_s[0]],
                                ymin=min(np.min(tt_e),np.min(tt_s)),
                                ymax=max(np.max(tt_e),np.max(tt_s)),
                                linestyles="dashed",
                                colors="black")
                ax["abs"].plot(abs_e,tt_e,color="chartreuse")
                ax["abs"].plot(abs_s,tt_s,color="plum")
                ax["abs"].set_xlabel("abscisse curviligne (px)")
                ax["abs"].set_ylabel("t (frame)")
                fig.suptitle(f"Chevauchement - ({e.index}-{s.index})")
                fig.savefig(self.output_dir+"Branches/"+f"chevauchement_{e.index}-{s.index}.jpg",dpi=300)
                plt.show()
                plt.close(fig)
            return
        """
        for b,e,s in tqdm(liste_chevauchements_filtree):
            visionnage(b,e,s,show=False)
        """
        return liste_chevauchements_filtree
    
    def treat_overlap(self, bp, be, bs)->bool:
        """
        Treat the current overlap by fusing be and bs and deconnecting them 
        from bp.
        """
        if be.index == bs.index:
            print("Attention, entrée == sortie")
            print(bp,be,bs)
            return False
        noeuds = np.array(list(self.g.nodes))
        
        #Première étape : Identification des noeuds de connexion
        #Et création du nouveau noeud n_new
        ne = be.noeuds[-1]
        ns = bs.noeuds[ 0]
        """
        if (ns not in bp.noeuds) or (ne not in bp.noeuds):
            #Est-il vraiment si mal déclaré ?
            print("Attention, chevauchement mal déclaré.")
            print("Les noeuds de connexion ne sont pas"\
                  " dans la branche porteuse.")
            print(bp,be,bs)
            fig,ax = plt.subplots()
            fig.suptitle(f"{bp.index} - {be.index} - {bs.index}")
            ax.plot([dX[n] for n in bp.noeuds],
                    [dY[n] for n in bp.noeuds],
                    color="skyblue")
            ax.plot([dX[n] for n in be.noeuds],
                    [dY[n] for n in be.noeuds],
                    color="forestgreen")
            ax.plot([dX[n] for n in bs.noeuds],
                    [dY[n] for n in bs.noeuds],
                    color="mediumpurple")
            plt.show()
            plt.close(fig)
            #return False
        """
        xye = np.array([self.n2x[ne],self.n2y[ne]])
        xys = np.array([self.n2x[ns],self.n2y[ns]])
        te = self.n2t[be.noeuds[-2]]
        ts = self.n2t[bs.noeuds[1]]
        xy_new = (xye+xys)*.5
        t_new = (te+ts)*.5
        #On indique les noeuds des chevauchements par un index négatif
        n_new = min(np.min(noeuds),0)-1 
        self.n2x[n_new] = xy_new[0]
        self.n2y[n_new] = xy_new[1]
        self.n2t[n_new] = t_new
        be.n2x = self.n2x
        be.n2y = self.n2y
        be.n2t = self.n2t
        #be.n2x[n_new] = xy_new[0]
        #be.n2y[n_new] = xy_new[1]
        #be.n2t[n_new] = t_new
        #Création du nouveau noeud
        self.g.add_nodes_from([(n_new,{"x": xy_new[0],
                                       "y": xy_new[1],
                                       "t": t_new})])
        #Connection du nouveau noeud
        self.g.add_edge(be.noeuds[-2],n_new)
        self.g.add_edge(n_new,bs.noeuds[+1])
        #Suppression des anciens liens
        self.g.remove_edge(be.noeuds[-2],ne)
        self.g.remove_edge(ns,bs.noeuds[+1])
        #Correction des brindilles de connexion.
        brindille_e = next(b 
                           for b in self.brindilles 
                           if b.index == be.brindilles[-1])
        brindille_e.noeuds[-1] = n_new
        brindille_s = next(b
                           for b in self.brindilles 
                           if b.index == bs.brindilles[ 0])
        brindille_s.noeuds[ 0] = n_new
        brindille_s.inBranches = [be.index]
        #Juxtaposition des branches be et bs.
        be.brindilles = [*be.brindilles,*bs.brindilles]
        be.noeuds = [*be.noeuds[:-1],n_new,*bs.noeuds[1:]]
        be.ending = bs.ending
        #Enregistrement du chevauchement dans la branche be et
        #dans la branche bp.
        #Quelles infos garde-t-on ?
        cas_traite = {"t_entree": te,
                      "t_sortie": ts,
                      "n":n_new,
                      "xy": xy_new,
                      "branches": (bp.index,be.index)}
        be.list_overlap.append(cas_traite)
        bp.list_overlap.append(cas_traite)
        #Redirection de la branche bs sur be
        for c in bs.list_overlap:
            #Il faut que je copie tous les chevauchements de bs
            # dans be
            b1_idx,b2_idx = c["branches"]
            if b1_idx == bs.index:
                c["branches"] = (be.index,b2_idx)
                b_autre = next(b for b in self.branches
                               if b.index == b2_idx)
            else:
                c["branches"] = (b1_idx,be.index)
                b_autre = next(b for b in self.branches
                               if b.index == b1_idx)
            be.list_overlap.append(c)
        #Suppression de bs
        kbs = next(k for k,b in enumerate(self.branches)
                   if b.index == bs.index)
        self.branches.pop(kbs)
        del bs 
        return True

    def overlaps(self,thresholdL_diameterUnit=5,thresholdT_frame = SEUIL_LAT):
        """
        Detect, sort and treat all overlap of the network
        """
        #First let's estimate the mean growth speed
        vitesses = []
        for b in self.branches:
            times,vs = b.normes_vitesses()
            vitesses.append(vs)
        vitesses = [element 
                    for subliste in vitesses 
                    for element in subliste]
        vmean = np.mean(vitesses)
        list_overlap = self.detect_overlaps(vmean,
                                            seuilL_diameterUnit=thresholdL_diameterUnit,
                                            seuilT=thresholdT_frame)

        def sort_overlap(liste:list[tuple])->list:
            g_sort = nx.DiGraph()
            g_sort.add_nodes_from(range(len(liste)))
            #Graph initialisation
            for k, triplet1 in enumerate(liste):
                for k2,triplet2 in enumerate(liste):
                    if (triplet1[0].index == triplet2[2].index or
                        triplet1[1].index == triplet2[2].index):
                        g_sort.add_edge(k,k2)
            sorted_list = []
            while(True):
                kfree = [n for n,d in g_sort.in_degree() if d == 0]
                if kfree:
                    for k in kfree:
                        sorted_list.append(liste[k])
                    g_sort.remove_nodes_from(kfree)
                else:
                    return sorted_list
                
        sorted_overlap = sort_overlap(list_overlap)
        print(f"{len(sorted_overlap)} overlap(s) to treat")
        for k,(b,e,s) in tqdm(enumerate(sorted_overlap)):
            ok = self.treat_overlap(b,e,s)
            if not ok:
                print(f"\nError with overlap number {k}\n")
        return self

    """
    ===========================================================================
    Utilitaires
    """
    def save(self,suffix:str):
        """
        Save the network in gpickle format
        """
        file = self.output_dir+self.name+"_"+suffix+".gpickle"
        with open(file,"wb") as f:
            pickle.dump(self,f)
        return self
    
    def network_at(self, f):
        """
        Renvoie le sous réseau g_frame extrait du réseau entier self.g contenant
        uniquement les points vérifiant t<=f
        """
        g_frame = self.g.copy()
        toKeep = [n for n in g_frame.nodes if self.n2t[n]<=f]
        g_frame = g_frame.subgraph(max(nx.connected_components(g_frame.subgraph(toKeep)),
                                          key = len)).copy()
        return g_frame
    
    def chaines_croissance(self):
        chaines = []
        for t in range(self.start+1,self.end+1):
            subg = self.g.subgraph((n for n in self.g.nodes if self.n2t[n]==t)).copy()
            subg_done = self.g.subgraph((n for n in self.g.nodes if self.n2t[n] < t)).copy()
            composants = nx.connected_components(subg)
            for composant in composants:
                composant = self.g.subgraph(composant).copy()
                if sum(1 for n,d in composant.degree() if d == 1 or d == 3)%2 == 0:
                    #On trouve la racine
                    degres1 = [n for n,d in composant.degree if d == 1]
                    racines = [r for r in composant.nodes
                               if np.any([v in subg_done.nodes 
                                          for v in self.g.neighbors(r)])]
                    if len(racines) == 1:
                        racine = racines[0]
                    else:
                        #Il faut trouver la bonne racine en regardant laquelle est connecté 
                        # à la partie la plus récente
                        t_raccord = np.array([next(self.n2t[v] for v in self.g.neighbors(r) 
                                                   if v in subg_done) 
                                              for r in racines]) 
                        t_raccord = t - t_raccord 
                        racine = racines[np.argmin(t_raccord)]
                    #On crée autant de chemin qu'il reste d'apex différents.
                    for a in degres1:
                        if a != racine:
                            chaines.append(nx.shortest_path(subg,source = racine, target =a))
        return chaines
    
    def growth_vectors(self):
        """
        Return all growth vectors
        """
        positions, growth = [], []
        for b in self.branches:
            pos, vit = b.positions_vitesses()
            positions.append(pos)
            growth.append(vit)
        #Applatissement
        positions = [p for pos in positions for p in pos]
        growth = [v for vit in growth for v in vit]
        return positions, growth
    
    def path_length(self, path:list[int])->float:
        """
        Return the length along a node path.
        Doesn't check if the path is a valid in the network.
        """
        pos = np.array([[self.n2x[n],self.n2y[n]] for n in path])
        L = np.sum(np.sqrt(((pos[1:,:]-pos[:-1,:])**2).sum(axis=1)))
        return L

    def neighborhood(self)->dict[int,dict]:
        """
        Return for each branch the list of incomming  and outgoing branches
        in the format:
        {b.index: {'in':  [b1,b2,...],
                   'out': [b3,b4,...]}
        }
        """
        #Initialisation:
        neighbors = {b.index: {'in': [],'out': []}
                     for b in self.branches}
        ends = {n: [] for n,d in self.g.degree if d != 2}
        for b in self.branches:
            ends[b.noeuds[+0]].append(b)
            ends[b.noeuds[-1]].append(b)
        for b in self.branches:
            #Pour chaque branche
            bidx = b.index
            for n in (n for n,d in self.g.degree(b.noeuds) if d != 2):
                #Pour chaque noeud de la branche
                voisines = ends[n]
                #On regarde les branches qui partagent ce noeud
                for bv in (v for v in voisines if v.index != bidx):
                    if bv.noeuds[0] == n:
                        #Si elle commence ici alors c'est une sortie
                        neighbors[bidx]['out'].append(bv)
                    elif bv.noeuds[-1] == n:
                        #Si elle se finit ici alors c'est une entrée
                        neighbors[bidx]['in'].append(bv)
        return neighbors
    
    def fuse(self,b1,b2):
        """
        Fuse branch b2 into branch b1 and then delete b2
        """
        if b1.noeuds[-1] != b2.noeuds[0]:
            raise ValueError(f"Branch {b2.index} is not a prolongation of {b1.index}")
        b1.noeuds = [*b1.noeuds,*b2.noeuds[1:]]
        b1.brindilles = [*b1.brindilles,*b2.brindilles]
        b1.ending = b2.ending
        b1.list_overlap = [*b1.list_overlap,*b2.list_overlap]
        for n in b1.noeuds:
            b1.n2x[n] = self.n2x[n]
            b1.n2y[n] = self.n2y[n]
            b1.n2t[n] = self.n2t[n]
        kb2 = next(k for k,b in enumerate(self.branches)
                   if b.index == b2.index)
        self.branches.pop(kb2)
        del b2
        return self

    def image_at(self, f:int):
        """
        Return the image corresponding to frame f open as numpy array
        """
        image = np.asarray(Image.open(self.imgs.format(f)))
        return image
    
    def prune_d2(self)->nx.Graph:
        """
        Return the networkx graph with only nodes that are not allways a degree 2 node.
        """
        pruned_g = self.g.copy()
        for b in self.branches:
            tt = np.array(b.t)
            dt = tt[1:]-tt[:-1]
            toKeep = np.array([self.g.degree(n)!=2 for n in b.noeuds])
            toKeep[:-1] = np.logical_or(toKeep[:-1],dt!=0)
            #Maintenant il 'suffit' de reconnecter entre 2 toKeep True consécutifs
            n0 = b.noeuds[0]
            for n,boolean in zip(b.noeuds[1:],toKeep[1:]):
                if boolean:
                    pruned_g.add_edge(n0,n)
                    n0 = n 
            pruned_g.remove_nodes_from(np.array(b.noeuds)[~toKeep])
        return pruned_g

    def find_loops_at(self,time:float)->list[list[int]]:
        """
        Find all loops at time 'time' in the reseau
        """
        g_t = fct.prune_upgraded(self.network_at(time))
        #Filtrage des noeuds de d1.
        d1 = [n for n,d in g_t.degree() if d ==1]
        while len(d1):
            g_t.remove_nodes_from(d1)
            d1 = [n for n,d in g_t.degree() if d ==1]
        g_directed = nx.DiGraph(g_t)
        unit_vectors = {}
        for u,v in g_t.edges:
            vector = np.array([self.n2x[v]-self.n2x[u],self.n2y[v]-self.n2y[u]])
            length = np.sqrt(np.dot(vector,vector))
            unit_vectors[(u,v)] = vector/length
            unit_vectors[(v,u)] = -vector/length

        def angle(vec1,vec2):
            vec1_orth = np.array([-vec1[1],vec1[0]])
            return np.arctan2(np.dot(vec2,vec1_orth),np.dot(vec2,vec1))

        def find_left(g_dir:nx.DiGraph,edge:tuple[int]):
            """
            Given an incoming edge (u,v) return the left most outgoing edge
            at node v that is not (v,u)
            """
            u,v = edge
            potentials = [(v,w) for w in g_dir.neighbors(v) if w != u]
            if len(potentials)==1:
                return potentials[0]
            uv = unit_vectors[(u,v)]
            thetas = [angle(uv,unit_vectors[pot]) for pot in potentials]
            return potentials[np.argmax(thetas)]
        loops = []
        edges = list(g_directed.edges)
        while len(edges):
            u,v = edges.pop()
            path = [u,v]
            while path[-1] != path[0]:
                v,w = find_left(g_directed,(u,v))
                path.append(w)
                u,v = v,w
            loops.append(path)
            g_directed.remove_edges_from([(x,y) for x,y in zip(path[:-1],path[1:])])
            edges = list(g_directed.edges)
        #il faut virer la boucle correspondant à l'extérieur (supposé la plus longue. Est-ce bien vrai ?)
        loops = sorted(loops,key=lambda loop:len(loop))
        if len(loops)==0:
            return []
        loops.pop()
        edge2nodes = nx.get_edge_attributes(g_t,"nodes")
        for i,loop in enumerate(loops):
            loop_extended = []
            for u,v in zip(loop[:-1],loop[1:]):
                nodes = edge2nodes.get((min(u,v),max(u,v)),-1)
                if type(nodes) is int:
                    nodes = edge2nodes.get((max(u,v),min(u,v)))

                if np.any([x not in [nodes[0],nodes[-1]] for x in (u,v)]):
                    print("Problème mes extrémités ne sont pas aux extrémités !")
                    print(u,nodes,v)
                if nodes[-1] == u and nodes[0] == v:
                    nodes = nodes[::-1]
                loop_extended.extend(nodes[:-1])
            loop_extended.append(v)
            loops[i] = loop_extended
        return loops

    def loop_graph_at(self,t:float=-1)->nx.Graph:
        """ 
        Return the loop graph at time t.
        If t<0 (by default) return the final dual_graph
        """
        t = t if t>0 else self.end
        loops = sorted(self.find_loops_at(t),key=lambda loop: len(loop))
        nodes = list(range(len(loops)))
        #Dual graph creation
        loop_g = nx.Graph()
        loop_g.add_nodes_from(nodes)
        node2loop = {i:loop for i,loop in enumerate(loops)}
        for i in nodes[:-1]:
            loopi = [n for n in node2loop[i] if self.g.degree(n)>2]#Pas besoin de checker tous les noeuds
            for j in nodes[i+1:]:
                loopj = [n for n in node2loop[j] if self.g.degree(n)>2]
                if np.any([n in loopj for n in loopi]):
                    loop_g.add_edge(i,j)
        x = {i : np.mean([self.n2x[n] for n in loop]) for i,loop in enumerate(loops)}
        y = {i : np.mean([self.n2y[n] for n in loop]) for i,loop in enumerate(loops)}
        node2loop = {i:loop for i,loop in enumerate(loops)}
        nx.set_node_attributes(loop_g,x,"x")
        nx.set_node_attributes(loop_g,y,"y")
        nx.set_node_attributes(loop_g,node2loop,"nodes")
        return loop_g

    def branch_graph_at(self, t:int):
        """
        Return the branch graph of the network at time t.
        The node of the branch graph are the branches of the network. 
        The edge (u,v) exist in the branch graph if branch u and v share a node.
        """
        existing_branches = [b for b in self.branches if b.get_tstart()<=t]
        branch_graph = nx.Graph()
        branch_graph.add_nodes_from((b.index for b in existing_branches))
        index_apex ={b.index: b.get_apex_at(t,index=True)[0] for b in existing_branches}
        for i1,b1 in enumerate(existing_branches[:-1]):
            b1_nodes = b1.noeuds[:index_apex[b1.index]+1]
            for i2,b2 in enumerate(existing_branches[i1+1:],start=i1+1):
                if np.any([n1 in b2.noeuds[:index_apex[b2.index]+1] for n1 in b1_nodes]):
                    branch_graph.add_edge(b1.index,b2.index)
        nx.set_node_attributes(branch_graph,
                               {b.index:b.abscisse_curviligne()[index_apex[b.index]]
                                for b in existing_branches},
                                name="length")
        return branch_graph
    
    def manual_branch_correction(self,branch1,branch2):
        """
        Prolongate branch1 with branch2, therefore branch2 should start with the ending node of branch1.
        """
        if branch2.noeuds[0]!= branch1.noeuds[-1]:
            raise ValueError(f"Branch {branch2.index} is not a valid candidate to prolongate branch {branch1.index}")
        branch1.noeuds.extend(branch2.noeuds[1:])
        branch1.list_overlap.extend(branch2.list_overlap)
        branch1.brindilles.extend(branch2.brindilles)
        branch1.n2x.update(branch2.n2x)
        branch1.n2y.update(branch2.n2y)
        branch1.n2t.update(branch2.n2t)
        branch1.ending = branch2.ending
        kbranch2 = next(k for k,b in enumerate(self.branches)
                        if b.index == branch2.index)
        self.branches.pop(kbranch2)
        return self

    def classification_nature_branches(self,threshold:float):
        """
        Classify each branch according to the latency before branching
        """
        for b in self.branches:
            b.nature = "Apical" if b.get_tstart()-b.t[0] <= threshold else "Lateral"
        return self
    
    """
    ===========================================================================
    Conversion
    """

    def convert4Fricker(self,path2file:str,suffix:str=""):
        """ 
        Conversion function to convert my object 'reseau' to an exploitable .txt 
        file as requested by Nicolas Fricker.
        Create 2 files, one listing all nodes and their coordinates, one containing 
        the adjacency matrix. 
        """
        noeuds = np.array(list(self.g.nodes))
        x = np.array([self.n2x[n] for n in noeuds])
        y = np.array([self.n2y[n] for n in noeuds])
        t = np.array([self.n2t[n] for n in noeuds])
        coordinates = np.rec.fromarrays([noeuds,x,y,t],names=["n","x","y","t"],
                                        dtype=[('n',int),("x",float),('y',float),("t",int)])
        b_index = []
        us = []
        vs = []
        for b in self.branches:
            bi = b.index
            for u,v in zip(b.noeuds[:-1],b.noeuds[1:]):
                b_index.append(bi)
                us.append(u)
                vs.append(v)
        b_index = np.array(b_index)
        us = np.array(us)
        vs = np.array(vs)    
        branches = np.rec.fromarrays([b_index,us,vs],names=["b_i","u","v"],
                                     dtype=[("b_i",int),("u",int),("v",int)])
        coordinates.tofile(path2file+f"{self.name}_coordinates{suffix}.txt")
        branches.tofile(path2file+f"{self.name}_branches{suffix}.txt")
        return self
    
    def convert2txt(self,path2file:str)->nx.Graph: 
        """
        Convert the network to a txt file of the form:
        U,V,XU,YU,TU,XV,YV,TV,B
        With (U,V) the edges X_,Y_,T_ the coordinates of the point and B the 
        index of the corresponding branch.
        In the header, informations can be found about the source and its position.
        """
        g_prune = self.prune_d2()
        data = []
        for b in self.branches:
            noeuds= [n for n in b.noeuds if n in g_prune]
            for u,v in zip(noeuds[:-1],noeuds[1:]):
                data.append([u,v,self.n2x[u],self.n2y[u],self.n2t[u],self.n2x[v],self.n2y[v],self.n2t[v],b.index])
        data = np.array(data)
        np.savetxt(path2file,data,fmt="%d",
                   header=f"#Spore = {self.source} XS = {int(self.n2x[self.source])}, YS = {int(self.n2y[self.source])}\n"+
                           "#U,V,XU,YU,TU,XV,YV,TV,B",
                   delimiter=",")
        return self
    """
    ===========================================================================
    Visualisation
    """
    def show_at(self,t:float, ax)->None:
        """
        Draw the network at the instant 't' on the axe 'ax'
        """
        degree2size = {1:10,2:2,3:14}
        degree2color = {1:"cyan",2:"slategrey",3:"orangered"}
        subg = self.g.subgraph((n for n in self.g.nodes if self.n2t[n]<= t))
        nx.draw(subg,pos={n:(self.n2x[n],self.n2y[n])
                          for n in subg.nodes},
                ax=ax,
                node_size=[degree2size.get(d,20) 
                           for n,d in subg.degree],
                node_color=[degree2color.get(d,"red") 
                            for n,d in subg.degree])
        return None

    @property
    def times(self)->np.ndarray:
        """
        Return the instant from self.start to self.end
        """
        times = np.arange(self.start,self.end+1)
        return times
    
    @property
    def Nbranches(self)->np.ndarray:
        """
        Return Nbranches the total number of branches at time t 
        for t ranging from self.start to self.end
        """
        N_t0 = np.array([b.get_tstart() for b in self.branches])
        Nbranches = np.array([np.sum(N_t0<=t) for t in self.times])
        return Nbranches
    
    @property
    def total_length(self)->np.ndarray:
        """
        Return total_length the length of the reseau
        for t ranging from self.start to self.end
        """
        L_edges = np.sqrt([(self.n2x[u]-self.n2x[v])**2+
                        (self.n2y[u]-self.n2y[v])**2
                        for u,v in self.g.edges])
        t_edges = np.array([max(self.n2t[u],self.n2t[v]) 
                            for u,v in self.g.edges])
        total_length = np.array([np.sum(L_edges[np.where(t_edges<=t)])
                                 for t in self.times])
        return total_length


"""
===========================================================================
Brindilles
"""
class Brindille():
    """
    Définition de la classe des brindilles.
    """
    def __init__(self,
                index:int,
                noeuds:list[int],
                n2x:dict[int,float],
                n2y:dict[int,float],
                n2t:dict[int,float],
                inBranches:list = [],
                confiance:float = 0):
        self.index = index #index de cette brindille
        self.noeuds = noeuds 
        self.n2x = n2x
        self.n2y = n2y 
        self.n2t = n2t 
        self.inBranches = inBranches #liste des index des branches
                                        #contenant cette brindille
        self.confiance = confiance
    
    def __repr__(self) -> str:
        repr = f"Brindille {self.index} - {len(self.noeuds)} noeuds"
        return repr
    
    def abscisse_curviligne(self)->np.ndarray:
        """
        Renvoie la liste des abscisses curvilignes des noeuds de la branche
        """
        pos = np.array([[self.n2x[n],self.n2y[n]] for n in self.noeuds])
        abscisse = np.sqrt(np.sum((pos[1:,:]-pos[:-1,:])**2,axis=-1))
        abscisse = np.cumsum(abscisse)
        abscisse = np.insert(abscisse,0,0)
        return abscisse
    
    def get_tstart(self)->float:                                        
        """
        Renvoie la coordonnée t correspondant au début de la brindille.
        """
        tstart = self.n2t[self.noeuds[1]]
        return tstart

    def get_tend(self)->float:
        """
        Renvoie la coordonnée t correspondant à la fin de la brindille.
        """
        tt = [self.n2t[n] for n in self.noeuds]
        tend = np.max(tt)
        return tend

    def is_growing_at(self, t:float)->bool:
        """
        Renvoie si oui ou non la branche est en train de croître 
        à l'instant t passé en argument.
        """
        return self.get_tstart()<=t<=self.get_tend()
    
    def detection_latence(self, seuil:int = 4)->bool:
        """
        Départ avec latence -> True 
        Départ sans latence -> False
        """
        bLatence = bool(self.get_tstart()-self.n2t[self.noeuds[0]] < seuil)
        return bLatence
    
    def unit_vector(self, end, r = np.inf):
        """
        Calcule le vecteur unitaire au niveau de l'extrémité spécifiée 
        mesuré avec un rayon r.
        """
        abscisse = self.abscisse_curviligne()
        if end not in [self.noeuds[0],self.noeuds[-1]]:
            raise ValueError(f"{end} is not an end of twig {self.index}")
        if self.noeuds[0] == end:
            kstop = next((k for k,s in enumerate(abscisse) if s > r),-1)
            u1 = end
            u2 = self.noeuds[kstop]
        else:
            abscisse = reversed(abscisse - abscisse[-1])
            kstop = next((len(self.noeuds)-k-1 for k,s in enumerate(abscisse) 
                        if s < -r),0)
            u1 = self.noeuds[kstop]
            u2 = end
        vect = np.array([self.n2x[u2]-self.n2x[u1],
                            self.n2y[u2]-self.n2y[u1]])
        unit_vect = vect/np.linalg.norm(vect)
        return unit_vect
    
    def reverse(self):
        """
        Reverse the twig.
        """
        self.noeuds.reverse()
        self.confiance = - self.confiance
        return self
    
    def calcul_confiance(self, seuil:float)->float:
        """
        Calcule la confiance dans l'orientation de la brindille
        """
        tt = np.array([self.n2t[n] for n in self.noeuds])
        dtt = tt[1:]-tt[:-1]
        #Incrément
        dtt = np.where(dtt>0, 1, dtt)
        #Décrément
        dtt = np.where(dtt<0,-1, dtt)
        #dtt = np.where(np.abs(dtt) != 1, 0, dtt)
        nP = np.sum(np.abs(dtt)+dtt)//2
        nM = np.sum(np.abs(dtt)-dtt)//2
        self.confiance = (nP-nM)/(nP+nM)*np.tanh((nP+nM)/seuil) if nP+nM else 0.
        return self.confiance
    
    def get_apex_at(self, t:float, index:bool = False)->int:
        """
        Return the current 'apex' on the twig|branch at time t
        if t>tEnd then the apex is the last node of the twig|branch
        if index == True then return also the index of apex in b.noeuds 
        """
        tt = [self.n2t[n] for n in self.noeuds]
        if self.get_tstart()>t:
            raise ValueError("La branche n'est pas encore apparu à ce moment.")
        iapex = next((i for i,ti in enumerate(tt[1:])
                    if ti > t),
                    len(tt)-1)
        apex = self.noeuds[iapex]
        if index:
            return (iapex,apex)
        return apex

    def get_all_apex(self)->tuple[list[int],list[int]]:
        """
        Return all the successive apex at the different growth time
        """
        tstart = int(self.get_tstart())
        tend = int(self.get_tend())
        time = list(range(tstart,tend+1))
        tt = self.t
        nnoeuds = len(self.noeuds)
        ntime = len(time)
        apex = []
        i=0
        it = 0
        while i < nnoeuds-1:
            if tt[i+1]>time[it]:
                apex.append(self.noeuds[i])
                it += 1
            else:
                i += 1
        time = list(range(tstart,tend+1))
        apex.append(self.noeuds[-1])
        while len(apex)<ntime:#Strange, should only occur with branches/twigs made of 2 nodes
            apex.append(self.noeuds[-1])#It's a fix but not a good one...
        return time, apex
    
    @property
    def x(self)->list[float]:
        """
        Return the list of x coordinate of each node in twig.nodes
        """
        return [self.n2x[n] for n in self.noeuds]
    
    @property
    def y(self)->list[float]:
        """
        Return the list of y coordinate of each node in twig.nodes
        """
        return [self.n2y[n] for n in self.noeuds]
    
    @property
    def t(self)->list[float]:
        """
        Return the list of t coordinate of each node in twig.nodes
        """
        return [self.n2t[n] for n in self.noeuds]
    
    @property
    def s(self)->np.ndarray:
        """
        Return the list of s, arc length of each node in twig.nodes
        """
        return self.abscisse_curviligne()

    @property
    def theta(self,radius:float = 7)->np.ndarray:
        """
        Return the list of theta, the orientation of the twig.
        Orientation is absolute.
        """
        x = np.array(self.x)
        y = np.array(self.y)
        thetas = np.zeros_like(x)
        #Cas général
        radius_squared = radius*radius
        for i,n in enumerate(self.noeuds):
            x0,y0 = x[i],y[i]
            r2 = (x-x0)*(x-x0)+(y-y0)*(y-y0)
            im = next((i-j for j,rsq in enumerate(r2[i::-1])if rsq>radius_squared ),
                    0)
            ip = next((j for j,rsq in enumerate(r2[i:],start=i) if rsq>radius_squared),
                      -1)
            im = next((i-j for j,rsq in enumerate(r2[i::-1])if rsq>radius_squared ),
                      0)
            thetas[i] = np.arctan2(y[ip]-y[im],x[ip]-x[im])
        return np.unwrap(thetas)

"""
===========================================================================
Branches
"""
class Branche(Brindille):
    """
    Définition de la classe secondaire des branches.
    Hérite de la classe secondaire des brindilles.
    """
    def __init__(self,
                    index:int, #int, index de la branche
                    noeuds:list[int], #Liste des noeuds de la branche
                    n2x:dict[int,float],
                    n2y:dict[int,float],
                    n2t:dict[int,float],
                    brindilles:list[int], #Liste des index des brindilles
                    nature:str, #nature de la branche : Apical,Lateral,Initial,...
                    ending:str, #raison de l'arret de croissance : d1, Fusion ?
                    list_overlap:list = None #Liste des chevauchements
                    ):
        self.index = index
        self.noeuds = noeuds
        self.n2x = n2x
        self.n2y = n2y
        self.n2t = n2t
        self.brindilles = brindilles
        self.nature = nature
        self.ending = ending
        self.list_overlap = list_overlap if list_overlap else []
    
    def __repr__(self) -> str:
        repr = f"Branche {self.index} - {len(self.noeuds)} noeuds"
        return repr

    def grow(self,brindille):
        """
        Prolongation de la branche par la brindille.
        """
        self.noeuds = [*self.noeuds,*brindille.noeuds[1:]]
        self.brindilles.append(brindille.index)
        return self

    def normes_vitesses(self, seuil_lat = 4):
        """
        Calcule et renvoie les normes des vitesses de la branche et la liste des
        instants correspondant.
        seuil_lat permet de ne pas prendre en compte les vitesses lentes
        due à une latence.
        Renvoie ([],[]) s'il n'est pas possible de définir une vitesse sur la branche
        """
        times,vitesses = self.vecteurs_vitesses(seuil_lat = seuil_lat)
        vs = []
        if len(vitesses):
            vs = np.sqrt(np.sum(vitesses*vitesses,axis=1))
        return times,vs

    def vecteurs_vitesses(self, seuil_lat = 4):
        """
        Calcule les vecteurs vitesses de croissance de la branche.
        Renvoie sous la forme d'un tuple la liste des instants t correspondant 
        et la liste des vecteurs vitesses.
        Renvoie [] s'il n'est pas possible de définir une vitesse sur la 
        branche.
        """
        vecteursV = []
        times = []
        pos = np.array([[self.n2x[n],self.n2y[n]] for n in self.noeuds])
        abscisse = self.s
        tt = np.array(self.t)
        index = [i for i,t in enumerate(tt[:-1]) if t != tt[i+1]]
        #index : liste des index où il y a variation de t
        if index:
            index.append(len(tt)-1)
            for i0,i in zip(index[:-1],index[1:]):
                deltaT = tt[i]-tt[i0]
                deltaS = abscisse[i]-abscisse[i0]
                deltaPos = pos[i,:]-pos[i0,:]
                if 0<deltaT<seuil_lat:
                    normedPos = np.linalg.norm(deltaPos)
                    direction = deltaPos/normedPos if normedPos else deltaPos
                    vecteursV.append(deltaS/deltaT*direction)
                    times.append((tt[i0]+tt[i])*.5)
        times = np.array(times)
        vecteursV = np.array(vecteursV)
        return times,vecteursV

    def positions_vitesses(self):     
        """
        Renvoie la liste des vecteurs vitesses et des positions 
        correspondantes de la branche.
        """
        dX,dY,dT = self.n2x,self.n2y,self.n2t
        pos = np.array([[dX[n],dY[n]] for n in self.noeuds])
        tstart = int(self.get_tstart())
        tend = int(self.get_tend())
        _,apex = self.get_all_apex()
        apex.insert(0,self.noeuds[0])
        temps = np.array([t-0.5 for t in range(tstart,tend+1)])
        if len(apex)<2:
            return np.empty(shape=(1,3)),np.empty(shape=(1,2))
        positions = np.zeros(shape=(len(apex)-1,3))
        vecteursV = np.zeros(shape=(len(apex)-1,2))
        positions[:,2] = np.arange(tstart,tend+1)-.5
        abscisse = self.abscisse_curviligne()
        n2i = {n:i for i,n in enumerate(self.noeuds)}
        positions_apex = np.array([pos[n2i[a]] for a in apex])
        positions[:,:2] = (positions_apex[:-1]+positions_apex[1:])*.5
        vecteursV = (positions_apex[1:]-positions_apex[:-1])
        cart_normes = np.linalg.norm(vecteursV,axis=1)
        curv_normes = np.array([abscisse[n2i[a1]]-abscisse[n2i[a2]]
                                for a1,a2 in zip(apex[1:],apex[:-1])])
        filtre = np.where(cart_normes>0)
        vecteursV[filtre,0] *= curv_normes[filtre]/cart_normes[filtre]
        vecteursV[filtre,1] *= curv_normes[filtre]/cart_normes[filtre]
        #Moyenne roulante 
        #positions = (positions[1:,:]+positions[:-1,:])/2
        #vecteursV = (vecteursV[1:,:]+vecteursV[:-1,:])/2
        return positions,vecteursV
    
    def apex(self)->list[int]:
        """
        Renvoie la liste des apex successifs de la branche
        """
        temps = [self.n2t[n] for n in self.noeuds]
        apex = [self.noeuds[i] for i in range(1,len(self.noeuds)-1) if temps[i] < temps[i+1]]
        apex.append(self.noeuds[-1])
        return apex


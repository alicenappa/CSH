import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#Imports needed
import numpy as np #For everything                              version 
import matplotlib.pyplot as plt #For visualisation              version 
import networkx as nx #For the graph                            version 
from pickle import load as pload #For loading the gpickle file  version 
from PIL import Image #For image management                     version 
from tqdm import tqdm #For nice loading bars                    version 
from declaration_tutti import load, treat_a_new_reseau, load_from_scratch
from fct_utils import prune,xy2t
from fct_analyse import arbre_genealogique_branches,color_by_time
#Our part:
from Reseau import Reseau,Brindille,Branche 
from scipy.optimize import curve_fit
from IPython.display import Image


def vision_des_branches_alice(reseau, t, rayon, demi_angle, show):
    """
    Renvoie à l'instant t si le champ de vision des branches présentes est occupé ou libre
    sous la forme d'un dictionnnaire {b.index: isFree?}
    Le champ de vision est un cone d'angle 2*'demi_angle' et de rayon 'rayon'.
    """
    isFree = {}
    obstacles = {}  # Dictionary to store obstacles for each branch
    branches = [b for b in reseau.branches if b.get_tstart() < t <= b.get_tend()]
    noeuds_actifs = [n for n in reseau.g.nodes if reseau.n2t[n] <= t]
    n2index = {n: i for i, n in enumerate(noeuds_actifs)}
    index2n = {i: n for i, n in enumerate(noeuds_actifs)}
    n2branch = {n: b for b in reseau.branches for n in b.noeuds}
    xx = np.array([reseau.n2x[n] for n in noeuds_actifs])
    xmin, xmax = np.min(xx) - reseau.diameter, np.max(xx) + reseau.diameter
    yy = np.array([reseau.n2y[n] for n in noeuds_actifs])
    ymin, ymax = np.min(yy) - reseau.diameter, np.max(yy) + reseau.diameter
    maillages = np.zeros(shape=(int((xmax - xmin) // rayon) + 1,
                                int((ymax - ymin) // rayon) + 1,
                                len(noeuds_actifs)),
                         dtype=bool)
    imax, jmax, kmax = maillages.shape
    
    def xy2ij(x, y):
        """
        given x,y coordinates of a node, find the i,j coordinates in the matrix 'maillages'
        """
        i = int((x - xmin) // rayon)
        j = int((y - ymin) // rayon)
        return i, j
    
    # Initialistation du maillage
    pos = {n: (reseau.n2x[n], reseau.n2y[n]) for n in noeuds_actifs}
    for k, n in enumerate(noeuds_actifs):
        i, j = xy2ij(*pos[n])
        maillages[i, j, k] = True
    
    g = nx.subgraph(reseau.g, noeuds_actifs).copy()
    
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw(g, pos=pos, ax=ax, node_size=2)
        ax.imshow(reseau.image_at(t), cmap="Greys")
    
    for b in branches:
        if len(b.noeuds)> 25: 
            color = "black"
            apex = b.get_apex_at(t)
            i, j = xy2ij(*pos[apex])
            
            # Find neighbors
            noeuds_potentiels = np.zeros_like(noeuds_actifs, dtype=bool)
            for im in range(max(i - 1, 0), min(i + 2, imax)):
                for jm in range(max(j - 1, 0), min(j + 2, jmax)):
                    noeuds_potentiels = np.where(maillages[im, jm, :], True, noeuds_potentiels)
            
            noeuds_potentiels[n2index[apex]] = False
            filtre = np.where(noeuds_potentiels)
            
            distance = np.sqrt((xx[filtre] - reseau.n2x[apex]) ** 2 +
                               (yy[filtre] - reseau.n2y[apex]) ** 2)
            
            try:
                pre_apex = b.get_apex_at(t - 1)
            except ValueError:
                pre_apex = b.noeuds[0]
            
            direction = np.array([reseau.n2x[apex] - reseau.n2x[pre_apex],
                                  reseau.n2y[apex] - reseau.n2y[pre_apex]])
            norm =  np.linalg.norm(direction) 
            
            if norm == 0: 
                continue
                #norm = 0.00000001
            
            direction = direction / norm 
            theta = np.arctan2(direction[1], direction[0])
            direction_ort = np.array([direction[1], -direction[0]])
            
            if np.any(distance < rayon):
                # We check for angle
                axes = np.array([xx[filtre] - reseau.n2x[apex],
                                 yy[filtre] - reseau.n2y[apex]]).T
    
                angle = np.arctan2(np.dot(axes, direction_ort),
                                   np.dot(axes, direction))
    
                test = np.logical_and(distance <= rayon,
                                      np.logical_and(angle <= demi_angle,
                                                     angle >= -demi_angle))
    
                isFree[b.index] = not np.any(test)
    
                if not isFree[b.index]:
                    obstacle_indices = np.where(test)[0]
    
                    if len(obstacle_indices) > 0:
                        temp =  [index2n[filtre[0][i]] for i in obstacle_indices]
                        obstacles[b.index] = {
                            'obstacles': temp ,
                            'distances': distance[test].tolist() , 
                            'direction' : direction      ,           #direzione rispetto ad un sdr assoluto al "centro" 
                            'apex' : apex
                             
                        }
                        
                        
            obstacles_branches= {}
            for index, value in obstacles.items(): 
                obstacles_branches[index] = []
                temp = value['obstacles']
                if temp != None:
                    for j in temp: 
                        f = n2branch[j]
                        subject = reseau.branches[index]
                        if f != subject:
                            obstacles_branches[index].append(f)
                        
                        
            if show:
                color = "green" if isFree[b.index] else "red"
                ax.plot([reseau.n2x[apex] + t * rayon * direction[0] for t in (0, 1)],
                        [reseau.n2y[apex] + t * rayon * direction[1] for t in (0, 1)],
                        color=color, ls='--')
                ax.plot([reseau.n2x[apex] + t * rayon * np.cos(theta + demi_angle) for t in (0, 1)],
                        [reseau.n2y[apex] + t * rayon * np.sin(theta + demi_angle) for t in (0, 1)],
                        color=color, ls='--')
                ax.plot([reseau.n2x[apex] + t * rayon * np.cos(theta - demi_angle) for t in (0, 1)],
                        [reseau.n2y[apex] + t * rayon * np.sin(theta - demi_angle) for t in (0, 1)],
                        color=color, ls='--')
                ax.plot([reseau.n2x[apex] + rayon * np.cos(theta + alpha) for alpha in np.linspace(-demi_angle, +demi_angle, 10)],
                        [reseau.n2y[apex] + rayon * np.sin(theta + alpha) for alpha in np.linspace(-demi_angle, +demi_angle, 10)],
                        color=color, ls='--')
        
            if show:
                ax.set_xlim([xmin - rayon, xmax + rayon])
                ax.set_ylim([ymin - rayon, ymax + rayon])
                fig.savefig(reseau.output_dir + f"Branches/collision_{t:02d}.jpg")
                plt.show()
            
    def extract_obstacles_branches(obstacles):     # funzione che crea il dizionario 'Branch :{ obstacles , distances}
 
        n2branch = {n: b for b in reseau.branches for n in b.noeuds}
        obstacles_branches = {}
        for index, b in obstacles.items():    #index cicla sui branches in questione quindi 1, 771, etc. 
            nodes = b['obstacles']    #lista dei nodi in obstacles 
            # lista delle distanze a cui corrisponde ciascuno di quei nodi 
            
            branches = []
            distancess= []
            mother =  reseau.branches[index] 
            for i , node in enumerate(nodes):
                OBS = n2branch[node]
                if str(mother)  != str(OBS) :
                    distances = b['distances']
                    directions = b['direction']
                    branches.append(OBS)
                    distancess.append( distances[i])
                    obstacles_branches[index] = {
                        'branches': branches,
                        'distances': distancess ,
                         'directions' : directions   }
        return obstacles_branches 
    def find_closest_obstacle():
        closest_obstacles = {}
        obstacle_branches = extract_obstacles_branches(obstacles)
        
        for index in obstacle_branches.keys(): 
            branches = obstacle_branches[index]['branches']
            distances = obstacle_branches[index]['distances'] 
            directions= obstacle_branches[index]['directions'] 
            if branches and distances:
                
                min_distance_index = distances.index(min(distances)) 
         
                closest_branch = branches[min_distance_index]
                closest_distance = distances[min_distance_index]
                closest_node = obstacles[index]['obstacles'][min_distance_index] 
                
                closest_obstacles[index] = {
                    'branch': closest_branch,
                    'distance': closest_distance ,
                    'direction' : directions , 
                    'node' : closest_node
                }
        
        return closest_obstacles
    closest_obstacles = find_closest_obstacle()
    obstacles_branches = extract_obstacles_branches(obstacles)
    return isFree, obstacles , obstacles_branches ,closest_obstacles












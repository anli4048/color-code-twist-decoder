import networkx as nx
import itertools as itt
import matplotlib.pyplot as plt
import numpy as np
import pymatching as pym
import copy

class color_code:
    """
    Class used to construct triangular color code patches, add twists, 
    and generate and decode bit-flip errors.
    """

    e = [np.array([[0],[1],[-1]]),np.array([[-1],[0],[1]]),np.array([[1],[-1],[0]])]

    def is_on_boundary (self, c):
        if type(c) == tuple:
            c = np.array(list(c))
        if type(c) == list:
            c = np.array(c)
        for k in range(0,3):
            if c[k] == int((self.d-1)/2):
                return True
        return False
    
    def is_in_bounds (self, c):
        if type(c) == tuple:
            c = np.array(list(c))

        for k in range(0,3):
            if c[k] > int((self.d-1)/2):
                return False
        return True
    
    def find_neighbor_of_checks (self, checks):
        neighbors = set()
        for check in checks:
            neighbors = neighbors.union(set(nx.neighbors(self.G,check)))
        return neighbors

    def find_common_neighbor_of_checks (self, checks):
        neighbors = set()
        is_first_set = True
        for check in checks:
            if is_first_set:
                neighbors = set(nx.neighbors(self.G,check))
                is_first_set = False
            else:
                neighbors = neighbors.intersection(set(nx.neighbors(self.G,check)))
        return neighbors
    
    def find_check_neighbours (self,check):
        return set(self.find_neighbor_of_checks(self.find_neighbor_of_checks([check]))).difference(set([check]))
    
    def find_complementary_colors (cols):
        comp_cols = []
        for c in ['blue','red','green']:
            if not c in cols:
                comp_cols.append(c)
            if comp_cols == ['blue','green']:
                return ['green','blue']
        return comp_cols
        
    def get_node_data (self, c):   
        if type(c) == tuple:
            c = np.array(list(c))

        #get position
        x_dir = np.array([2,0])
        y_dir = np.array([1,-np.sqrt(3)])
        pos = tuple(((c[0]*x_dir + c[1]*y_dir)).tolist())

        #get color and type
        single_color_id = {0:'green',1:'red',2:'blue'}
        double_color_id = {0:'magenta',1:'cyan',2:'yellow'}
        for i in range(0,len(self.e)):
            if np.array_equal(np.mod(c+self.e[i].flatten(),np.array([3,3,3])),np.array([0,0,0])):
                return {"type":"check","color":single_color_id[i],"pos":pos,"is_boundary":False,"faulty":False}
            if np.array_equal(np.mod(2*c+self.e[(i+1)%3].flatten()+self.e[(i+2)%3].flatten(),np.array([3,3,3])),np.array([0,0,0])):
                return {"type":"check","color":double_color_id[i],"pos":pos,"is_boundary":False,"faulty":False}
        return {"type":"error","color":'black',"pos":pos,"is_boundary":False,"faulty":False}
    
    def reposition_node (self, pos, col):
        # figure out the boundary equation in cartesian space
        color_nums = {'red': 0,
                      'blue': 1,
                      'green': 2}
        norms = {0: np.array([1/2, 1/(2*np.sqrt(3))]),
                 2: np.array([0, -1/np.sqrt(3)]),
                 1: np.array([-1/2, 1/(2*np.sqrt(3))])
                }
        b_num = ((self.d - 1)/2 - color_nums[col])%3
        # boundary equation is norm[col] dot [x  y] = (d-1)/2
        # we need to reflect across the boundary eq
        # lets rotate coordinates so the boundary is vertical
        # then reflect and then rotate back
        t = -np.arctan(norms[b_num][0]/norms[b_num][1])
        rot_mat = np.array([[np.cos(-t), -np.sin(-t)],[np.sin(-t), np.cos(-t)]])
        inv_rot_mat = np.array([[np.cos(t), -np.sin(t)],[np.sin(t), np.cos(t)]])
        rot_pos = rot_mat @ np.array(list(pos))[:, np.newaxis]
        y0_rot = (self.d-1)/(2*(norms[b_num][1]*np.cos(t)-norms[b_num][0]*np.sin(t)))
        new_rot_pos = np.array([[1, 0],[0,-1]]) @ rot_pos + np.array([[0],[2*y0_rot]])
        new_pos = inv_rot_mat @ new_rot_pos
        return tuple(new_pos.flatten().tolist())
            

    def __init__ (self, d, p, bound = 'triangle'):
        """
        Constructs a matching graph for a color code with triangular boundaries

        Params
        ------
        `d` is the distance of the code or the number of qubits along its boundary.
        `p` is the bit-flip error rate of the code
        """
        self.d = d
        self.bound = bound
        self.p = p

        self.shared_checks = list()
        self.long_pairs = list()
        self.check_pairs = list()

        if bound == 'triangle':
            assert d % 2 == 1, "d should be odd"

            # generate a coarse range of check coordinates
            coarse_range = range(-d,d+1)
            coords_temp = list((np.array([[x],[y],[-(x+y)]]) for x in coarse_range for y in coarse_range))
            coords_list = list()
            for c in coords_temp:
                in_bounds = True
                for k in range(0,3):
                    if c[k,0] > int((d-1)/2):
                        in_bounds = False
                if not in_bounds:
                    continue
                coords_list.append(c)
            coords_list = list(map(lambda c: tuple(c.flatten().tolist()), coords_list))
            coords_list = list(map(lambda c: (c,self.get_node_data(c)), coords_list))
        
        G = nx.Graph(bound=self.bound,d=self.d)
        G.add_nodes_from(coords_list)
        check_nodes = [check for check,data in G.nodes(data=True) if data["type"]=="check"]

        def get_edges_from (check):
            check_arr = np.array(list(check))
            return [(check,tuple((check_arr + (-1)**j*self.e[i].flatten()).tolist())) 
                    for i in range(0,len(self.e)) for j in range(0,2)
                    if self.is_in_bounds(check_arr + (-1)**j*self.e[i].flatten())
                    ]

        edges = []
        [edges.extend(get_edges_from(check)) for check in check_nodes]
        G.add_edges_from(edges)

        # add three boundary vertices
        for col in ['red','green','blue']:
            node_pos = self.reposition_node((0,0),col)
            node_col = col
            node_type = 'check'
            G.add_nodes_from([((None,col,col),{"type":node_type,"color":node_col,"pos":node_pos,"is_boundary":True,"faulty":False})])   
        
        # connect boundary vertices to boundary
        # for every error connected to only two checks, 
        # connect to the remaining color boundary vertex
        # if one check, connect to two boundary vertex
        for error,data in G.nodes(data=True):
            if data["type"]=="error":
                if len(list(G.neighbors(error))) == 2:
                    neighbors = list(G.neighbors(error))
                    col_dict = nx.get_node_attributes(G,'color')
                    remaining_col = color_code.find_complementary_colors((col_dict[neighbors[0]],col_dict[neighbors[1]]))[0]
                    G.add_edges_from([(error,(None,remaining_col,remaining_col))])
                if len(list(G.neighbors(error))) == 1:
                    neighbors = list(G.neighbors(error))
                    col_dict = nx.get_node_attributes(G,'color')
                    remaining_cols = color_code.find_complementary_colors([col_dict[neighbors[0]]])
                    G.add_edges_from([(error,(None,remaining_cols[0],remaining_cols[0]))])
                    G.add_edges_from([(error,(None,remaining_cols[1],remaining_cols[1]))])

        self.G = G
    
    def move_errors_off_wall_preproc (self):
        """
        Calculates preprocessing information to be passed to `move_errors_off_wall`. 
        This allows for time saves when running repeated shots.

        Returns
        ------
        A dictionary of ways to move errors off the wall to be passed to 
        `move_errors_off_wall`.
        """
        twist_checks = [(check,data) for check,data in self.G.nodes(data=True) if (data["type"]=="check" and not(data["color"] in ['red','blue','green']))]
        mapping = dict()                       
        
        
        for c in self.shared_checks:
            # move shared checks off wall
            # do this via error connecting shared check and two main color checks
            correction = list()
            for err in nx.neighbors(self.G,c):
                is_valid = True
                for adj_check in nx.neighbors(self.G,err):
                    if not (nx.get_node_attributes(self.G,"color")[adj_check] in ['red','green','blue']):
                        is_valid = False
                        break
                if is_valid:
                    correction.append(err)
                    break
            
            mapping[c] = tuple(correction)
        
        self.move_errors_mapping = mapping
        return mapping

    def move_errors_off_wall (self, mapping = None):
        """
        Moves shared color checks in the syndrome off domain walls.

        Params
        `mapping` provides preprocessing information. It is an optional 
        argument; as long as you have run `move_errors_off_wall_preproc`
        the `color_code` object will have a copy which it can access.
        """
        if mapping == None:
            mapping = self.move_errors_mapping
        
        is_faulty_g = nx.get_node_attributes(self.G,"faulty")
        is_faulty_g_read = nx.get_node_attributes(self.G,"faulty")
        correction = {node: False for node,data in self.G.nodes(data=True) if data["type"]=="error"}

        for c in self.shared_checks:
            if is_faulty_g_read[c]:
                for error in mapping[c]:
                    is_faulty_g[error] = not is_faulty_g[error]
                    correction[error] = not correction[error]

        nx.set_node_attributes(self.G,is_faulty_g,"faulty")
        return [node for node in correction.keys() if correction[node]]

    def generate_combined_restricted_lattice (self, color):
        """
        Constructs a combined restricted lattice with some shared color

        Params
        ------
        `color` is the shared color between the two halves of the combined 
        restricted lattice

        Returns
        ------
        A `networkx.Graph` storing the combined restricted lattice. A copy of 
        this is also stored with the `color_code` object.
        """
        assert (color == 'blue' or color == 'red' or color == 'green'), "invalid color"
        check_nodes = dict()
        for c in ['blue','red','green']:
            check_nodes[c] = [(check,data) for check,data in self.G.nodes(data=True) if (data["type"]=="check" and data["color"]==c and not (check[0]==None and data["color"]==color))]
        # create two copies of restricted lattice nodes
        # def add_lattice_color_attribute (node,c):
        #     node[1]["lattice_color"] == c

        rG = nx.Graph(color=color)
        for c in ['blue','red','green']:
            if not (c == color):
                temp_check_nodes = []
                for node in check_nodes[c]:
                    temp_check_nodes.append((tuple(list(node[0]).__add__(color_code.find_complementary_colors([c,color]))),node[1]))
                rG.add_nodes_from(temp_check_nodes)
        for c in color_code.find_complementary_colors([color]):
            temp_check_nodes = []
            for node in check_nodes[color]:
                    temp_check_nodes.append((tuple(list(node[0]).__add__([c])),node[1]))
            rG.add_nodes_from(temp_check_nodes)

        # connect two nodes if they share an error mechanism
        for c in color_code.find_complementary_colors([color]):
            for check1,data1 in check_nodes[c]:
                for check2,data2 in check_nodes[color]:
                    lattice_color = color_code.find_complementary_colors([c,color])[0]
                    if len(set(self.G.neighbors(check1)).intersection(set(self.G.neighbors(check2))))>0:
                        rG.add_edge(tuple(list(check1).__add__([lattice_color])),
                                    tuple(list(check2).__add__([lattice_color]))
                                    )
                        
        # connect two nodes along the shared boundary if they share an error mechanism
        close_bound_nodes = []
        far_bound_nodes = []
        for check,data in check_nodes[color_code.find_complementary_colors([color])[0]]:
            for i in range(0,3):
                if check[i] == (self.d-1)/2:
                    close_bound_nodes.append(check)
        for check,data in check_nodes[color_code.find_complementary_colors([color])[1]]:
            for i in range(0,3):
                if check[i] == (self.d-3)/2:
                    far_bound_nodes.append(check)

        for check1 in close_bound_nodes:
            for check2 in far_bound_nodes:
                lattice_color1 = color_code.find_complementary_colors([color,check1])[0]
                lattice_color2 = color_code.find_complementary_colors([color,check2])[1]
                if len(set(self.G.neighbors(check1)).intersection(set(self.G.neighbors(check2))))>0:
                    rG.add_edge(tuple(list(check1).__add__([lattice_color2])),
                                tuple(list(check2).__add__([lattice_color1]))
                                )
                    
        # connect boundary vertices with zero weight edge
        is_boundary = nx.get_node_attributes(rG,"is_boundary")
        boundary_nodes = list()
        for node in rG.nodes:
            if is_boundary[node]:
                boundary_nodes.append(node)
        
        rG.add_edge(*tuple(boundary_nodes),mult=-1)
        
        def find_nodes (check):
            nodes = list()
            for c in rG:
                if (c[0],c[1],c[2]) == check:
                    nodes.append(c)
            return nodes
        
        # add multicolor domain wall checks
        check_nodes = [(check,data) for check,data in self.G.nodes(data=True) if data["type"]=="check"]
        for check,data in check_nodes:
            if not(data['color'] in ['red','blue','green']):
                for col in color_code.find_complementary_colors([rG.graph['color']]):
                    rG.add_nodes_from([((*check,col),data)])
        
        # connect all skip nodes
        for c in self.shared_checks:
            degree_c = len(self.find_neighbor_of_checks([c]))
            for rc in find_nodes (c):
                neighbors = list(nx.neighbors(rG,rc))
                for a in neighbors:
                    for b in neighbors:
                        if a != b:
                            central_node = c
                            current_node = a[0:3]
                            target_node = b[0:3]
                            visited_nodes = set([current_node])

                            num_errors = 0
                            
                            while current_node != target_node:
                                neighbour_node = list((self.find_check_neighbours(central_node).intersection(self.find_check_neighbours(current_node))).difference(visited_nodes))[0]
                                num_errors += 1
                                visited_nodes.add(neighbour_node)
                                current_node = neighbour_node
                            
                            if min(num_errors,degree_c-num_errors) % 2 == 0:
                                rG.add_edge(a,b,mult=2)
            
        # connect all special edges
        col_map = nx.get_node_attributes(rG,"color")
        for e in self.check_pairs:
            rG.add_edge(find_nodes(e[0])[0],find_nodes(e[1])[0],mult=3)

        for e in self.long_pairs:
            u = find_nodes(e[0])[0]
            v = None
            for node in find_nodes(e[1]):
                if u[3] == node[3]:
                    v = node
            rG.add_edge(u,v,mult=4)
        
        # connect all twists
        for check,data in check_nodes:
            if not(data['color'] in ['red','blue','green']):
                comp_colors = color_code.find_complementary_colors([rG.graph['color']])
                u = (*check,comp_colors[0])
                v = (*check,comp_colors[1])
                rG.add_edge(u,v,mult=-1)
        
        # remove all nodes on boundary
        for c in self.shared_checks:
            for rc in find_nodes(c):
                if col_map[rc] in ['red','green','blue']:
                    rG.remove_node(rc)


        self.rG = rG
        return rG
    
    def insert_twist (self, start, end):
        """
        Inserts a domain wall in the matching graph with endpoints as given. Do not
        intersect two domain walls; this may lead to unexpected behaviour.

        Params
        ------
        `start` is a tuple providing the coordinates for an endpoint of the domain wall.
        `end` is a tuple providing the coordinates for another endpoint of the domain wall.

        Returns
        ------
        A tuple of data regarding checks around the twist. This data is stored
        in the `color_code` object so the end user does not need 
        """
        # bad things will probably happen if two intersecting twists are inserted
        # check start and end share coordinate and are checks
        # double every qubit along the interval from start to end, color accordingly
        assert (start[0]==end[0] or start[1]==end[1] or start[2]==end[2]), "Coordinates do not line up"
        assert not (start == end), "Coordinates are degenerate"
        assert (start[0]+start[1]+start[2]==0), "Start coordinate invalid"
        assert (end[0]+end[1]+end[2]==0), "End coordinate invalid"
        assert start in self.G.nodes(), "Start coordinate not in lattice"
        assert end in self.G.nodes(), "End coordinate not in lattice"
        assert nx.get_node_attributes(self.G,"type")[start] == "check", "Start coordinate is not check"
        assert nx.get_node_attributes(self.G,"type")[end] == "check", "End coordinate is not check"

        # find shared coordinate
        shared_coord = -1
        for i in range(0,3):
            if start[i]==end[i]:
                shared_coord = i
        other_coord = (shared_coord+1)%3 

        # get all nodes along defect
        defect_errors = []
        for node,data in self.G.nodes(data=True):
            if node[shared_coord] == start[shared_coord]:
                if data["type"] == "error":
                    if node[other_coord] > start[other_coord] and node[other_coord] < end[other_coord]:
                        defect_errors.append((node,data))
                    elif node[other_coord] < start[other_coord] and node[other_coord] > end[other_coord]:               
                        defect_errors.append((node,data))

        offset = [0,0,0]
        offset[shared_coord] = 0.4
        offset[other_coord] = -offset[shared_coord]/2
        offset[(other_coord+1)%3] = -offset[shared_coord]/2
        offset = np.array(offset)
        
        # duplicate nodes and rewire connections
        offset_errors1 = dict([(err,tuple((np.array(list(err))+offset).tolist())) for err,data in defect_errors])
        self.G = nx.relabel_nodes(self.G, offset_errors1)

        offset_errors2 = [(tuple((np.array(list(err))-offset).tolist()),data) for err,data in defect_errors]  
        self.G.add_nodes_from(offset_errors2) 

        def get_close (x,ys):
            for y in ys:
                b_match = True
                for i in range(0,3):
                    if not x[i]-y[0][i] < 0.01:
                        b_match = False
                if b_match:
                    return y[0]


        new_edges = [(get_close(tuple((np.array(list(u))-2*offset).tolist()),offset_errors2),v) for u,v in self.G.edges() if u in offset_errors1.values()]   
        new_edges.extend([(u,get_close(tuple((np.array(list(v))-2*offset).tolist()),offset_errors2)) for u,v in self.G.edges() if v in offset_errors1.values()])
        self.G.add_edges_from(new_edges)

        color_1 = None
        color_2 = None
        shared_color = None
        check_pairs = set()

        for u in offset_errors1.values():
            for v in list(nx.neighbors(self.G,u)):
                if v[shared_coord] > u[shared_coord]:
                    color_1 = nx.get_node_attributes(self.G,"color")[v]
            break
        
        for u,data in offset_errors2:
            for v in list(nx.neighbors(self.G,u)):
                if v[shared_coord] < u[shared_coord]:
                    color_2 = nx.get_node_attributes(self.G,"color")[v]
            break

        shared_color = color_code.find_complementary_colors([color_1,color_2])[0]

        for u in offset_errors1.values():
            pair = list()
            for v in list(nx.neighbors(self.G,u)): 
                if nx.get_node_attributes(self.G,"color")[v] == color_2:
                    self.G.remove_edge(u,v)
                    pair.append(v)
                if nx.get_node_attributes(self.G,"color")[v] == color_1:
                    pair.append(v)
            if pair[0] < pair[1]:
                check_pairs.add((pair[0],pair[1]))
            else:
                check_pairs.add((pair[1],pair[0]))
        
        for u,data in offset_errors2:
            pair = list()
            for v in list(nx.neighbors(self.G,u)):      
                if nx.get_node_attributes(self.G,"color")[v] == color_1:
                    self.G.remove_edge(u,v)
                    pair.append(v)
                if nx.get_node_attributes(self.G,"color")[v] == color_2:
                    pair.append(v)
            if pair[0] < pair[1]:
                    check_pairs.add((pair[0],pair[1]))
            else:
                check_pairs.add((pair[1],pair[0]))

        # add plaquettes along boundary
        error_quads = list()
        for pair in check_pairs:
            quad = list()
            for u in offset_errors1.values():
                if len(set(pair).intersection(set(nx.neighbors(self.G,u)))) > 0:
                    quad.append(u)
            for u,data in offset_errors2:
                if len(set(pair).intersection(set(nx.neighbors(self.G,u)))) > 0:
                    quad.append(u)
            error_quads.append(quad)
        
        twist_color_map = {'red':'cyan', 'green':'magenta','blue':'yellow'}
        for quad in error_quads:
            twist_check = np.array([0,0,0])
            n = 0
            for err in quad:
                twist_check = twist_check + np.array(list(err))
                n = n + 1
            twist_check = tuple((twist_check/n).tolist())
            data = {'type':'check','color':twist_color_map[shared_color],'pos':(0,0),'faulty':False,'is_boundary':False}
            self.G.add_nodes_from([(twist_check,data)])
            for err in quad:
                self.G.add_edges_from([(err,twist_check)])

        pos = nx.get_node_attributes(self.G,"pos")
        for p in pos.keys():
            if p[0] != None:
                x_dir = np.array([2,0])
                y_dir = np.array([1,-np.sqrt(3)])
                pos[p] = tuple(((p[0]*x_dir + p[1]*y_dir)).tolist())
        nx.set_node_attributes(self.G,pos,"pos")

        long_pairs = set()
        shared_checks = set()
        check_quads = list()
        for quad in error_quads:
            check_quad = set()
            for error in quad:
                for check in nx.neighbors(self.G,error):
                    if nx.get_node_attributes(self.G,"color")[check] in ['red','green','blue']:
                        check_quad.add(check)
                    if nx.get_node_attributes(self.G,"color")[check] == shared_color:
                        shared_checks.add(check)
            check_quads.append(check_quad)
        

        for quad in check_quads:
            adj_pairs = list()
            for u in quad:
                if nx.get_node_attributes(self.G,"color")[u] == shared_color:
                    for v in quad:
                        if nx.get_node_attributes(self.G,"color")[v] != shared_color:
                            adj_pairs.append((u,v))

            for p in adj_pairs:
                adj_p_checks = self.find_neighbor_of_checks(list(self.find_common_neighbor_of_checks(p))).difference(set(p))
                u = None
                v = None
                for check in adj_p_checks:
                    if nx.get_node_attributes(self.G,"color")[check] in ['red','green','blue']:
                        u = check
                    else:
                        v = check
                long_pairs.add((u,v))
                
                

        self.check_pairs.extend(check_pairs)
        self.long_pairs.extend(long_pairs)
        self.shared_checks.extend(shared_checks)
        return check_pairs,long_pairs,shared_checks


    def draw_combined_restricted_lattice (self, rG, size, draw_weightless = False):
        combined_col = rG.graph['color']
        lattice_cols = color_code.find_complementary_colors([combined_col])
        pos_dict = dict()

        # sort node positions of restricted graph by lattice color
        for col in lattice_cols:
            pos_dict[col] = list()
        for node,data in rG.nodes(data=True):
            pos_dict[node[3]].append((node,data['pos']))

        # apply offset to node positions of second lattice color
        
        offset_nodes = list()
        for node,pos in pos_dict[lattice_cols[1]]:
            offset_nodes.append((node,self.reposition_node(pos, combined_col)))
        pos_dict[lattice_cols[1]] = offset_nodes
        graph_pos = dict(pos_dict[lattice_cols[0]]+pos_dict[lattice_cols[1]])
        

        # draw nodes
        def get_linewidth(check,data):
            if data["faulty"]:
                return 2
            else:
                return 0
            
        def edge_colors(data):
            if "marked" in data.keys():
                if data["marked"]:
                    return 'yellow'
            if "mult" in data.keys():
                if data["mult"] == -1:
                    return "black"
                if data["mult"] == 2:
                    return 'grey'
                if data["mult"] == 3:
                    return 'purple'
                if data["mult"] == 4:
                    return 'brown'
            return 'black'
        
        edgelist = list(rG.edges())


        edge_cols = [edge_colors(data) for u,v,data in rG.edges(data=True)]
        lwds = [get_linewidth(check,data) for check,data in rG.nodes(data=True)]
        check_node_color = [data["color"] for check,data in rG.nodes(data=True)]
        nodes = nx.draw_networkx_nodes(rG,graph_pos,node_size= size, nodelist = rG.nodes, node_color = check_node_color, node_shape='s',linewidths=lwds, edgecolors='k')
        nx.draw_networkx_edges(rG,graph_pos,edge_color=edge_cols)
        plt.axis('scaled')  
        # nodes.set_edgecolor('k')

    def modify_combined_restricted_lattice (self,rG):
        rG2 = copy.deepcopy(rG)

        # add nodes for checks along the boundary
        check_nodes = [(check,data) for check,data in self.G.nodes(data=True) if data["type"]=="check"]
        for check in self.shared_checks:
            check_data = None
            for node,data in self.G.nodes(data=True):
                if node == check:
                    check_data = data
            for col in color_code.find_complementary_colors([rG.graph['color']]):  
                rG2.add_nodes_from([((*check,col),check_data)])
        
        # connect nodes
        twist_checks = [(check,data) for check,data in self.G.nodes(data=True) if (data["type"]=="check" and not(data["color"] in ['red','blue','green']))]
        for check,data in twist_checks:
            for c in self.find_neighbor_of_checks(list(nx.neighbors(self.G,check))):
                in_check_pair = False
                for pair in self.check_pairs:
                    if c in pair:
                        in_check_pair = True
                if in_check_pair:
                    continue
                if c == check:
                    continue
                for col in color_code.find_complementary_colors([rG.graph['color']]):
                    rG2.add_edge((*check,col),(*c,col),marked=False)

        
        for check in self.shared_checks:
            for c in self.find_neighbor_of_checks(list(nx.neighbors(self.G,check))):
                if c == check:
                    continue
                for col in color_code.find_complementary_colors([rG.graph['color']]):
                    if (*c,col) in rG2.nodes():
                        if not (((*check,col),(*c,col)) in rG2.edges() or ((*c,col),(*check,col)) in rG2.edges()):
                            rG2.add_edge((*check,col),(*c,col),marked=False)

                

        
        # convert to second restricted lattice
        edges = copy.deepcopy(rG2.edges(data=True))
        for u,v,data in edges:
            if "mult" in data.keys():                        
                rG2.remove_edge(u,v)
        return rG2
    
    def mark_modified_restricted_lattice_prepoc (self,rG):
        mapping = {edge: None for edge in rG.edges()}

        edges = rG.edges(data=True)
        for u,v,data in edges:
            if "mult" in data.keys():
                if data["mult"] == 2:
                    u_neighbors = self.find_neighbor_of_checks(list(nx.neighbors(self.G,u[0:3]))).difference(set([u[0:3]]))
                    v_neighbors = self.find_neighbor_of_checks(list(nx.neighbors(self.G,v[0:3]))).difference(set([v[0:3]]))
                    shared_neighbors = list(u_neighbors.intersection(v_neighbors))
                    shared_neighbor = None
                    for neighbor in shared_neighbors:
                        if neighbor in self.shared_checks:
                            shared_neighbor = neighbor
                    
                    edge_list = list()

                    edge_list.append((u,(*shared_neighbor,u[3])))
                    edge_list.append((v,(*shared_neighbor,v[3])))
                    mapping[(u,v)] = tuple(edge_list)
                elif data["mult"] == 3:
                    u_neighbors = self.find_neighbor_of_checks(list(nx.neighbors(self.G,u[0:3]))).difference(set([u[0:3]]))
                    v_neighbors = self.find_neighbor_of_checks(list(nx.neighbors(self.G,v[0:3]))).difference(set([v[0:3]]))
                    first_u_neighbor = None
                    for u_neigh in u_neighbors:
                        if u_neigh in self.shared_checks and u_neigh in v_neighbors:
                            first_u_neighbor = u_neigh
                            break

                    second_u_neighbor = None
                    u_neighbors = self.find_neighbor_of_checks(list(nx.neighbors(self.G,first_u_neighbor[0:3]))).difference(set([first_u_neighbor[0:3]]))
                    for u_neigh in u_neighbors:
                        if not (nx.get_node_attributes(self.G,"color")[u_neigh] in ["red","green","blue"]) and u_neigh in v_neighbors:
                            second_u_neighbor = u_neigh
                            break

                    edge_list = list()

                    edge_list.append((u,(*first_u_neighbor,u[3])))
                    edge_list.append(((*first_u_neighbor,u[3]),(*second_u_neighbor,u[3])))
                    edge_list.append((v,(*first_u_neighbor,v[3])))
                    edge_list.append(((*first_u_neighbor,v[3]),(*second_u_neighbor,v[3])))
                    mapping[(u,v)] = tuple(edge_list)
                elif data["mult"] == 4:
                    all_u_neighbors = self.find_check_neighbours(u[0:3])

                    u_neighbors = [u_neigh for u_neigh in all_u_neighbors if (u_neigh in self.shared_checks)]
                    first_u_neighbor = None

                    for u_neigh in u_neighbors:
                        if v[0:3] in self.find_check_neighbours(u_neigh):
                            first_u_neighbor = u_neigh
                            break

                    edge_list = list()

                    edge_list.append((u,(*first_u_neighbor,u[3])))
                    edge_list.append(((*first_u_neighbor,u[3]),v))
                    
                    mapping[(u,v)] = tuple(edge_list)
                elif data["mult"] == -1:
                    mapping[(u,v)] = None
            else:
                edge_list = list()

                edge_list.append((u,v))
                mapping[(u,v)] = tuple(edge_list)
        self.mapping = mapping
        return mapping
    
    def mark_modified_restricted_lattice (self,rG2,rG,mapping = None):
        if mapping == None:
            mapping = self.mapping

        self.mark_errors_restricted_lattice(rG=rG2)
        marked_dict = {edge: False for edge in rG2.edges()}

        edges = rG.edges(data=True)
        for u,v,data in edges:
            if not data["marked"]:
                continue
            edge = None
            if (u,v) in mapping.keys():
                edge = (u,v)
            else:
                edge = (v,u)
            
            if mapping[edge] == None:
                continue
            for a,b in list(mapping[edge]):
                mod_edge = None
                if (a,b) in marked_dict.keys():
                    mod_edge = (a,b)
                else:
                    mod_edge = (b,a)
                marked_dict[mod_edge] = not marked_dict[mod_edge]
                
                

        nx.set_edge_attributes(rG2,marked_dict,"marked")

                
    
    def mark_modified_restricted_lattice_legacy (self,rG2,rG):
        self.mark_errors_restricted_lattice(rG=rG2)
        marked_dict = {edge: False for edge in rG2.edges()}

        edges = rG.edges(data=True)
        for u,v,data in edges:
            if "mult" in data.keys():
                if data["mult"] == 2:
                    if data["marked"]:
                        u_neighbors = self.find_neighbor_of_checks(list(nx.neighbors(self.G,u[0:3]))).difference(set([u[0:3]]))
                        v_neighbors = self.find_neighbor_of_checks(list(nx.neighbors(self.G,v[0:3]))).difference(set([v[0:3]]))
                        shared_neighbors = list(u_neighbors.intersection(v_neighbors))
                        shared_neighbor = None
                        for neighbor in shared_neighbors:
                            if neighbor in self.shared_checks:
                                shared_neighbor = neighbor
                        
                        if (u,(*shared_neighbor,u[3])) in marked_dict.keys():
                            marked_dict[(u,(*shared_neighbor,u[3]))] = not marked_dict[(u,(*shared_neighbor,u[3]))]
                        else:
                            marked_dict[((*shared_neighbor,u[3]),u)] = not marked_dict[((*shared_neighbor,u[3]),u)]

                        if (v,(*shared_neighbor,v[3])) in marked_dict.keys():
                            marked_dict[(v,(*shared_neighbor,v[3]))] = not marked_dict[(v,(*shared_neighbor,v[3]))]
                        else:
                            marked_dict[((*shared_neighbor,v[3]),v)] = not marked_dict[((*shared_neighbor,v[3]),v)]
                elif data["mult"] == 3:
                    if data["marked"]:
                        u_neighbors = self.find_neighbor_of_checks(list(nx.neighbors(self.G,u[0:3]))).difference(set([u[0:3]]))
                        v_neighbors = self.find_neighbor_of_checks(list(nx.neighbors(self.G,v[0:3]))).difference(set([v[0:3]]))
                        first_u_neighbor = None
                        for u_neigh in u_neighbors:
                            if u_neigh in self.shared_checks and u_neigh in v_neighbors:
                                first_u_neighbor = u_neigh
                                break

                        second_u_neighbor = None
                        u_neighbors = self.find_neighbor_of_checks(list(nx.neighbors(self.G,first_u_neighbor[0:3]))).difference(set([first_u_neighbor[0:3]]))
                        for u_neigh in u_neighbors:
                            if not (nx.get_node_attributes(self.G,"color")[u_neigh] in ["red","green","blue"]) and u_neigh in v_neighbors:
                                second_u_neighbor = u_neigh
                                break
                        
                        if (u,(*first_u_neighbor,u[3])) in marked_dict.keys():
                            marked_dict[(u,(*first_u_neighbor,u[3]))] = not marked_dict[(u,(*first_u_neighbor,u[3]))]
                        else:
                            marked_dict[((*first_u_neighbor,u[3]),u)] = not marked_dict[((*first_u_neighbor,u[3]),u)]
                        
                        if ((*first_u_neighbor,u[3]),(*second_u_neighbor,u[3])) in marked_dict.keys():
                            marked_dict[((*first_u_neighbor,u[3]),(*second_u_neighbor,u[3]))] = not marked_dict[((*first_u_neighbor,u[3]),(*second_u_neighbor,u[3]))]
                        else:
                            marked_dict[((*second_u_neighbor,u[3]),(*first_u_neighbor,u[3]))] = not marked_dict[((*second_u_neighbor,u[3]),(*first_u_neighbor,u[3]))]

                        if (v,(*first_u_neighbor,v[3])) in marked_dict.keys():
                            marked_dict[(v,(*first_u_neighbor,v[3]))] = not marked_dict[(v,(*first_u_neighbor,v[3]))]
                        else:
                            marked_dict[((*first_u_neighbor,v[3]),v)] = not marked_dict[((*first_u_neighbor,v[3]),v)]

                        if ((*first_u_neighbor,v[3]),(*second_u_neighbor,v[3])) in marked_dict.keys():
                            marked_dict[((*first_u_neighbor,v[3]),(*second_u_neighbor,v[3]))] = not marked_dict[((*first_u_neighbor,v[3]),(*second_u_neighbor,v[3]))]
                        else:
                            marked_dict[((*second_u_neighbor,v[3]),(*first_u_neighbor,v[3]))] = not marked_dict[((*second_u_neighbor,v[3]),(*first_u_neighbor,v[3]))]              
                elif data["mult"] == 4:
                    if data["marked"]:
                        all_u_neighbors = self.find_check_neighbours(u[0:3])
                        all_v_neighbors = self.find_check_neighbours(v[0:3])

                        u_neighbors = [u_neigh for u_neigh in all_u_neighbors if (u_neigh in self.shared_checks)]
                        v_neighbors = [v_neigh for v_neigh in all_v_neighbors if (v_neigh in self.shared_checks)]
                        first_u_neighbor = None
                        first_v_neighbor = None
                        second_u_neighbor = None
                        second_v_neighbor = None

                        for u_neigh in u_neighbors:
                            break_loop = False
                            for v_neigh in v_neighbors:
                                shared_double_neighbors = (self.find_check_neighbours(u_neigh).difference(set([u_neigh]))).intersection(self.find_check_neighbours(v_neigh).difference(set([v_neigh])))
                                if len(shared_double_neighbors) > 0:
                                    found_shared_neighbor = False
                                    for second_neighbor in shared_double_neighbors:
                                        if not (nx.get_node_attributes(self.G,"color")[second_neighbor] in ['red','green','blue']):
                                            second_u_neighbor = second_neighbor
                                            found_shared_neighbor = True
                                            break
                                    if found_shared_neighbor:
                                        second_v_neighbor = second_u_neighbor
                                        first_u_neighbor = u_neigh
                                        first_v_neighbor = v_neigh
                                        break_loop = True
                                        break
                            if break_loop:
                                break
                        
                        if (u,(*first_u_neighbor,u[3])) in marked_dict.keys():
                            marked_dict[(u,(*first_u_neighbor,u[3]))] = not marked_dict[(u,(*first_u_neighbor,u[3]))]
                        else:
                            marked_dict[((*first_u_neighbor,u[3]),u)] = not marked_dict[((*first_u_neighbor,u[3]),u)]
                        
                        if ((*first_u_neighbor,u[3]),(*second_u_neighbor,u[3])) in marked_dict.keys():
                            marked_dict[((*first_u_neighbor,u[3]),(*second_u_neighbor,u[3]))] = not marked_dict[((*first_u_neighbor,u[3]),(*second_u_neighbor,u[3]))]
                        else:
                            marked_dict[((*second_u_neighbor,u[3]),(*first_u_neighbor,u[3]))] = not marked_dict[((*second_u_neighbor,u[3]),(*first_u_neighbor,u[3]))]

                        if (v,(*first_v_neighbor,v[3])) in marked_dict.keys():
                            marked_dict[(v,(*first_v_neighbor,v[3]))] = not marked_dict[(v,(*first_v_neighbor,v[3]))]
                        else:
                            marked_dict[((*first_v_neighbor,v[3]),v)] = not marked_dict[((*first_v_neighbor,v[3]),v)]

                        if ((*first_v_neighbor,v[3]),(*second_v_neighbor,v[3])) in marked_dict.keys():
                            marked_dict[((*first_v_neighbor,v[3]),(*second_v_neighbor,v[3]))] = not marked_dict[((*first_v_neighbor,v[3]),(*second_v_neighbor,v[3]))]
                        else:
                            marked_dict[((*second_v_neighbor,v[3]),(*first_v_neighbor,v[3]))] = not marked_dict[((*second_v_neighbor,v[3]),(*first_v_neighbor,v[3]))]
            else:
                if data["marked"]:
                    if (u,v) in marked_dict.keys():
                        marked_dict[(u,v)] = not marked_dict[(u,v)]
                    else: 
                        marked_dict[(v,u)] = not marked_dict[(v,u)]
            
            nx.set_edge_attributes(rG2,marked_dict,"marked")

    def lift_restricted_lattice_prepoc (self,rG):
        # Make a mapping that takes a check and gives the edges incident of it
        incident_edge_mapping = dict()
        check_nodes = [node for node,data in self.G.nodes(data=True) if data["color"]==rG.graph["color"]]

        for check in check_nodes:
            incident_edges = set()
            for col in color_code.find_complementary_colors([rG.graph['color']]):
                for u,v,data in rG.edges((*check,col),data=True): 
                    incident_edges.add((u[0:3],v[0:3]))
            incident_edge_mapping[check] = tuple(incident_edges)

        # Make a mapping that takes a pair of edges and gives the errors contained inside
        edge_pair_mapping = dict()
        for check in incident_edge_mapping.keys():
            for edge1 in incident_edge_mapping[check]:
                for edge2 in incident_edge_mapping[check]:
                    pair = (edge1,edge2)
                    if pair[0] == pair[1]:
                        continue
                    
                    central_node = list(set(pair[0]).intersection(set(pair[1])))[0]
                    current_node = list(set(pair[0]).difference(set([central_node])))[0]
                    target_node = list(set(pair[1]).difference(set([central_node])))[0]
                    visited_nodes = set([current_node])

                    error_list = list()

                    while current_node != target_node:
                        neighbour_node = list((self.find_check_neighbours(central_node).intersection(self.find_check_neighbours(current_node))).difference(visited_nodes))[0]
                        error = list(self.find_common_neighbor_of_checks([central_node,current_node,neighbour_node]))[0]
                        error_list.append(error)
                        visited_nodes.add(neighbour_node)
                        current_node = neighbour_node
                    
                    if len(error_list) % 2 != 0 and len(self.find_neighbor_of_checks([central_node])) % 2 != 0:
                        error_list = list(set(self.find_neighbor_of_checks([central_node])).difference(set(error_list)))
                    edge_pair_mapping[(pair[0],pair[1])] = tuple(error_list)
                    edge_pair_mapping[(pair[1],pair[0])] = tuple(error_list)
        self.edge_pair_mapping = edge_pair_mapping

        # Make a mapping that finds the errors contained in a boundary edge
        boundary_edge_mapping = dict()
        for u,v,data in rG.edges(data=True):
            if u[3] != v[3]:
                check_neighbours = self.find_check_neighbours(u[0:3]).intersection(self.find_check_neighbours(v[0:3]))
                boundary_node = None
                for check in check_neighbours:
                    if check[0] == None:
                        boundary_node = check
                error = list(self.find_common_neighbor_of_checks([u[0:3],v[0:3],boundary_node]))[0]
                boundary_edge_mapping[(u,v)] = error
        self.boundary_edge_mapping = boundary_edge_mapping

        return edge_pair_mapping,boundary_edge_mapping

    def lift_restricted_lattice (self,rG,edge_pair_map=None,boundary_edge_map=None):
        if edge_pair_map == None:
            edge_pair_map = self.edge_pair_mapping
        if boundary_edge_map == None:
            boundary_edge_map = self.boundary_edge_mapping

        error_nodes = [node for node,data in self.G.nodes(data=True) if data["type"]=="error"]
        check_nodes = [node for node,data in self.G.nodes(data=True) if data["type"]=="check"]
        errors = {error: False for error in error_nodes}
        is_marked = nx.get_edge_attributes(rG,"marked")
        color_map = nx.get_node_attributes(self.G,"color")

        for check in check_nodes:
            if color_map[check] != rG.graph['color']:
                continue

            incident_edges = list()
            for col in color_code.find_complementary_colors([rG.graph['color']]):
                for u,v,data in rG.edges((*check,col),data=True):
                    if "marked" in data.keys():
                        if data["marked"]:  
                            if (u[0:3],v[0:3]) in incident_edges:
                                incident_edges.remove((u[0:3],v[0:3]))
                            else:
                                incident_edges.append((u[0:3],v[0:3]))

            incident_edge_range = range(0,int(len(incident_edges)/2))
            for i in incident_edge_range:
                for error in edge_pair_map[(incident_edges[2*i],incident_edges[2*i+1])]:
                    errors[error] = not errors[error]
        
        for u,v in boundary_edge_map.keys():
            b_marked = False
            if (u,v) in is_marked.keys():
                if is_marked[(u,v)]:
                    b_marked = True
            if (v,u) in is_marked.keys():
                if is_marked[(v,u)]:
                    b_marked = True
            if b_marked:
                errors[boundary_edge_map[(u,v)]] = not errors[boundary_edge_map[(u,v)]]
        
        # return set of decoded error nodes
        return errors            


    def extract_error_set (self):
        # for each error node if the node is marked add to set and then return set
        errors = set()
        is_faulty_g = nx.get_node_attributes(self.G,"faulty")
        error_nodes = [check for check,data in self.G.nodes(data=True) if data["type"]=="error"]
        for error in error_nodes:
            if is_faulty_g[error]:
                errors.add(error)
        return errors


    def combine_errors (self, errors, decoding):
        # return union difference intersection of errors and decoding
        return (set(errors).union(set(decoding))).difference((set(errors).intersection(set(decoding))))
    
    def check_logical (self, errors, decoding, G = None):
        if G==None:
            G = copy.deepcopy(self.G)
            self.wipe_errors_graph(G)

        self.make_errors(0,self.combine_errors(errors,decoding),G)
        self.mark_errors(G)
        check_nodes = [check for check,data in G.nodes(data=True) if data["type"]=="check"]
        is_faulty_g = nx.get_node_attributes(G,"faulty")
        for check in check_nodes:
            if is_faulty_g[check] and check[0] != None:
                self.wipe_errors_graph(G)
                return False
        self.wipe_errors_graph(G)
        return True

    def make_twist_loop_logicals (self, twist):
        # returns a loop logical surrounding a twist
        # twist is a tuple of the endpoints of a twist
        # at least two twists are needed for the loop logical to be nontrivial (in the bulk)
        # but we can nevertheless calculate it regardless

        # find shared checks on the twist
        twist_color = nx.get_node_attributes(self.G,"color")[twist[0]]
        shared_coordinate = None
        shared_checks = set()
        for i in range(0,3):
            if twist[0][i] == twist[1][i]:
                shared_coordinate = i
                break
        
        for node in [n for n,data in self.G.nodes(data=True) if data["color"]==twist_color and data["type"]=="check"]:
            if node[shared_coordinate]==twist[0][shared_coordinate]:
                next_coordinate = (shared_coordinate+1)%3
                if node[next_coordinate] <= max(twist[0][next_coordinate],twist[1][next_coordinate]):
                    if node[next_coordinate] >= min(twist[0][next_coordinate],twist[1][next_coordinate]):
                        shared_checks.add(node)

        # shared checks <- shared checks on the twist
        # start node <- neighbour check of some shared check on the twist
        # logical color <- color of current node
        # logical checks <- green check neighbours of elements of shared checks
        # current node <- start node
        # visited checks <- singleton set containing start node
        # while start node != current_node or the loop is in first iteration:
        #   for check in logical checks \ visited checks:
        #       if check and current node share two check neighbours:
        #           next node <- check
        #           break
        #   flip errors enclosed by current/next node and the two shared neighbours
        
        logical_color = color_code.find_complementary_colors([twist_color])[0]

        start_node = None
        for node in self.find_check_neighbours(twist[0]):
            if nx.get_node_attributes(self.G,"color")[node] == logical_color:
                start_node = node

        logical_checks = set()
        for shared_check in shared_checks:
            for neighbour in self.find_check_neighbours(shared_check):
                if nx.get_node_attributes(self.G,"color")[neighbour] == logical_color:
                    logical_checks.add(neighbour)
        
        first_iteration = True
        current_node = start_node
        visited_nodes = set([current_node])
        error_map = {error: False for error,data in self.G.nodes(data=True) if data["type"]=="error"}
        
        while current_node != start_node or first_iteration:
            if len(visited_nodes) == len(logical_checks):
                visited_nodes.remove(start_node)

            first_iteration = False
            next_node = None

            current_neighbours = list(nx.neighbors(self.G,current_node))
            for check in logical_checks.difference(visited_nodes):
                break_bool = False
                next_neighbours = list(nx.neighbors(self.G,check))
                for error1 in current_neighbours:
                    for error2 in next_neighbours:
                        if len(set(nx.neighbors(self.G,error1)).intersection(set(nx.neighbors(self.G,error2)))) >= 2:
                            error_map[error1] = not error_map[error1]
                            error_map[error2] = not error_map[error2]
                            next_node = check
                            break_bool = True
                            break
                    if break_bool:
                        break
                if break_bool:
                    break

            current_node = next_node
            visited_nodes.add(current_node)
        
        return [node for node in error_map.keys() if error_map[node]]

    def make_crossing_logical(self,twist):
        # returns a string logical connecting two boundaries via a twist

        color = nx.get_node_attributes(self.G,"color")[twist[0]]
        comp_colors = list(color_code.find_complementary_colors([color]))

        error_map = {error: False for error,data in self.G.nodes(data=True) if data["type"]=="error"}
        boundary_nodes = list()
        for col in comp_colors:
            boundary_nodes.append((None,col,col))
        
        twist_node = None
        twist_pair = list()
        shared_coordinate = None

        for i in range(0,3):
            if twist[0][i] == twist[1][i]:
                shared_coordinate = i
                break

        for node,data in self.G.nodes(data=True):
            if (not (data["color"] in ['red','green','blue'])) and data["type"]=="check":
                if np.abs(node[shared_coordinate] - twist[0][shared_coordinate]) < 0.001:
                    twist_node = node
                    break
        
        col_map = nx.get_node_attributes(self.G,"color")
        for col in comp_colors:
            for check in self.find_check_neighbours(twist_node):
                if col_map[check] == col:
                    twist_pair.append(check)

        assert len(twist_pair)==2
        for check in twist_pair:
            error = list(self.find_common_neighbor_of_checks([twist_node,check]))[0]
            error_map[error] = not error_map[error]                  

        def find_color_neighbours (check,col_map):
            color_neighbors = set()
            for error in nx.neighbors(self.G,check):
                check_neighbors = set(nx.neighbors(self.G,error)).difference(set([check]))

                corner_found = True
                for n in check_neighbors:
                    if n[0] != None:
                        corner_found = False
                if corner_found:
                    continue

                bridge_errors = set(self.find_common_neighbor_of_checks(check_neighbors)).difference(set([error]))
                bridge_error = list(bridge_errors)[0]
                color_neighbor = list(set(nx.neighbors(self.G,bridge_error)).difference(check_neighbors))[0]
                if col_map[color_neighbor] == col_map[check]:
                    color_neighbors.add(color_neighbor)
            return color_neighbors
        
        def find_bridging_errors (check1, check2):
            for error in nx.neighbors(self.G,check1):
                check_neighbors = set(nx.neighbors(self.G,error)).difference(set([check1]))

                corner_found = True
                for n in check_neighbors:
                    if n[0] != None:
                        corner_found = False
                if corner_found:
                    continue

                bridge_errors = set(self.find_common_neighbor_of_checks(check_neighbors)).difference(set([error]))
                bridge_error = list(bridge_errors)[0]
                color_neighbor = list(set(nx.neighbors(self.G,bridge_error)).difference(check_neighbors))[0]
                if color_neighbor == check2:
                    return (error,bridge_error)
        
        # MAKE GRAPH OF NODES OF TWIST COLOR
        for j in range(0,2):
            comp_col = comp_colors[j]
            node1 = boundary_nodes[j]
            node2 = twist_pair[j]

            assert col_map[node1] == comp_col
            assert col_map[node2] == comp_col

            color_graph = nx.Graph()
            for node,data in self.G.nodes(data=True):
                if data["color"] == comp_col:
                    color_graph.add_node(node)
            
            for node in color_graph.nodes():
                for neighbor in find_color_neighbours(node,col_map=col_map):
                    assert neighbor in color_graph.nodes(), "neighbour is {neigh}, col is {col}".format(neigh=neighbor,col=comp_col)
                    color_graph.add_weighted_edges_from([(node,neighbor,1)])
            
            path = nx.dijkstra_path(color_graph, node1, node2)

            for i in range(0, len(path)-1):
                errors = find_bridging_errors(path[i],path[i+1])
                for error in errors:
                    error_map[error] = not error_map[error]

        
        return [node for node in error_map.keys() if error_map[node]]
        

    def make_twist_string_logicals (self,twist_pair):
        # returns a string logical connecting two twists
        # twist_pair is a tuple of two twists - each twist is a tuple of the endpoints of a twist
        # at least two twists are needed for the loop logical to be nontrivial (in the bulk)
        # but we can calculate it regardless

        # activate two shared endpoint checks via all surrounding errors
        # connect shared checks with string
        twist_color = nx.get_node_attributes(self.G,"color")[twist_pair[0][0]]
        error_map = {error: False for error,data in self.G.nodes(data=True) if data["type"]=="error"}
        for twist in twist_pair:
            for error in nx.neighbors(self.G,twist[0]):
                error_map[error] = not error_map[error]
        
        def find_color_neighbours (check):
            color_neighbors = set()
            for error in nx.neighbors(self.G,check):
                check_neighbors = set(nx.neighbors(self.G,error)).difference(set([check]))

                corner_found = True
                for n in check_neighbors:
                    if n[0] != None:
                        corner_found = False
                if corner_found:
                    continue

                bridge_errors = set(self.find_common_neighbor_of_checks(check_neighbors)).difference(set([error]))
                bridge_error = list(bridge_errors)[0]
                color_neighbor = list(set(nx.neighbors(self.G,bridge_error)).difference(check_neighbors))[0]
                color_neighbors.add(color_neighbor)
            return color_neighbors
        
        def find_bridging_errors (check1, check2):
            for error in nx.neighbors(self.G,check1):
                check_neighbors = set(nx.neighbors(self.G,error)).difference(set([check1]))

                corner_found = True
                for n in check_neighbors:
                    if n[0] != None:
                        corner_found = False
                if corner_found:
                    continue

                bridge_errors = set(self.find_common_neighbor_of_checks(check_neighbors)).difference(set([error]))
                bridge_error = list(bridge_errors)[0]
                color_neighbor = list(set(nx.neighbors(self.G,bridge_error)).difference(check_neighbors))[0]
                if color_neighbor == check2:
                    return (error,bridge_error)
        
        # MAKE GRAPH OF NODES OF TWIST COLOR
        color_graph = nx.Graph()
        for node,data in self.G.nodes(data=True):
            if data["color"] == twist_color:
                color_graph.add_node(node)
        
        for node in color_graph.nodes():
            for neighbor in find_color_neighbours(node):
                color_graph.add_weighted_edges_from([(node,neighbor,1)])
        
        path = nx.dijkstra_path(color_graph, twist_pair[0][0], twist_pair[1][0])

        for i in range(0, len(path)-1):
            errors = find_bridging_errors(path[i],path[i+1])
            for error in errors:
                error_map[error] = not error_map[error]

        
        return [node for node in error_map.keys() if error_map[node]]

    
    def make_boundary_logical (self,color):
        # returns a string logical along a boundary
        # pick a boundary node - return all errors adjacent to it
        for node in self.G.nodes():
            if node == (None,color,color):
                return list(nx.neighbors(self.G,node))
    
    def check_trivial (self, errors,logicals):
        pass
        # need to check commutator with all logicals
        # logical 1 - logical connecting boundaries
        # logical 2 - loop around twist
        # logical 3 - connecting twists
        # these logicals generate the algebra of logicals
        # checking commutator is pretty simple in the bit flip picture - 
        # just check that the intersection of errors and logical is of even cardinality
        for logical in logicals:
            if bool(len(set(errors).intersection(logical))%2):
                return False
        return True


            


        
               
    def draw(self,size,graph=None):
        if graph == None:
            G = self.G
        else:
            G = graph
        pos=nx.get_node_attributes(G,'pos')
        check_nodes = [check for check,data in self.G.nodes(data=True) if data["type"]=="check"]
        error_nodes = [error for error,data in self.G.nodes(data=True) if data["type"]=="error"]

        check_node_color = [data["color"] for check,data in G.nodes(data=True) if data["type"]=="check"]
        error_node_color = ['yellow' if nx.get_node_attributes(G,'faulty')[error] else 'black' for error in error_nodes]
        def get_linewidth(check,data):
            if data["faulty"]:
                return 2
            else:
                return 0
        lwds = [get_linewidth(check,data) for check,data in G.nodes(data=True) if data["type"]=="check"]
        nodes = nx.draw_networkx_nodes(G,pos,node_size= size, nodelist = check_nodes, node_color = check_node_color, node_shape='s',linewidths=lwds,edgecolors='k')
        nx.draw_networkx_nodes(G,pos,node_size= size, nodelist = error_nodes, node_color = error_node_color, node_shape='o')
        nx.draw_networkx_edges(G,pos)
        plt.axis('scaled')
        # nodes.set_edgecolor('k')

    def make_errors (self,p,errors=None,G=None):
        # create random independent flip errors in matching graph
        if G==None:
            G = self.G
        if not (errors == None):
            error_nodes = [error for error,data in G.nodes(data=True) if data["type"]=="error"]
            is_faulty_g = nx.get_node_attributes(G,"faulty")
            for err in error_nodes:
                if err in errors:
                    is_faulty_g[err] = True
                else:
                    is_faulty_g[err] = False
            nx.set_node_attributes(G,is_faulty_g,"faulty")


        error_nodes = [error for error,data in self.G.nodes(data=True) if data["type"]=="error"]
        is_faulty_g = nx.get_node_attributes(self.G,"faulty")
        for err in error_nodes:
            if np.random.rand() < p:
                is_faulty_g[err] = True
        nx.set_node_attributes(self.G,is_faulty_g,"faulty")
        # print(nx.get_node_attributes(self.G,"faulty"))  
    
    def save_errors (self):
        faulty_error_nodes = [node for node,data in self.G.nodes(data=True) if (data["type"]=="error" and data["faulty"])]
        return faulty_error_nodes
    
    def mark_errors (self, G=None):
        # find corresponding syndrome
        if G==None:
            G = self.G
        check_nodes = [check for check,data in G.nodes(data=True) if data["type"]=="check"]
        is_faulty_g = nx.get_node_attributes(G,"faulty")
        for check in check_nodes:
            error_parity = 0
            for err in nx.neighbors(G, check):
                error_parity += int(is_faulty_g[err])
            error_parity = bool(error_parity%2)
            is_faulty_g[check] = error_parity

        nx.set_node_attributes(G,is_faulty_g,"faulty")
        # print(nx.get_node_attributes(self.rG,"faulty"))
    
    def mark_errors_restricted_lattice (self,rG=None):
        if rG==None:
            rG= self.rG
        check_nodes = [check for check,data in self.G.nodes(data=True) if data["type"]=="check"]
        col_map = nx.get_node_attributes(self.G,"color")
        is_faulty_g = nx.get_node_attributes(self.G,"faulty")
        is_faulty_rg = nx.get_node_attributes(rG,"faulty")
        for check in check_nodes:
            for col in ['red','green','blue']:
                r_check = tuple(list(check)+[col])
                if r_check in is_faulty_rg.keys():
                    is_faulty_rg[r_check]= is_faulty_g[check]
                    if not (col_map[check] in ['red','green','blue']):
                        break
        
        nx.set_node_attributes(rG,is_faulty_rg,"faulty")

    def wipe_errors (self):
        is_faulty_g = nx.get_node_attributes(self.G,"faulty")
        is_faulty_rg = nx.get_node_attributes(self.rG,"faulty")

        for node in is_faulty_g.keys():
            is_faulty_g[node] = False
        
        for node in is_faulty_rg.keys():
            is_faulty_rg[node] = False
        
        nx.set_node_attributes(self.G,is_faulty_g,"faulty")
        nx.set_node_attributes(self.rG,is_faulty_rg,"faulty")
    
    def wipe_errors_graph (self,G):
        is_faulty_g = nx.get_node_attributes(G,"faulty")

        for node in is_faulty_g.keys():
            is_faulty_g[node] = False
        
        nx.set_node_attributes(G,is_faulty_g,"faulty")
    
    def make_matching(self,p,rel_weights,auto_boundary=True):
        # set edge weights to w = log((1-p)/p) and probabilities to p
        rG = copy.deepcopy(self.rG)
        if not auto_boundary:
            is_boundary = nx.get_node_attributes(rG,"is_boundary")
            for node in rG.nodes():
                is_boundary[node] = False
            nx.set_node_attributes(rG,is_boundary,"is_boundary")

        weights = dict()
        probs = dict()
        mult = nx.get_edge_attributes(rG,"mult")
        for edge in rG.edges():
            if not (edge in mult.keys()):
                weights[edge] = -np.log(p)
                probs[edge] = 2*p*(1-p)
            elif mult[edge] == 2:
                weights[edge] = -rel_weights[0]*np.log(p)
                probs[edge] = p**2
            elif mult[edge] == 3:
                weights[edge] = -rel_weights[1]*np.log(p)
                probs[edge] = 2*p**2*(1-p)**2
            elif mult[edge] == 4:
                weights[edge] = -rel_weights[2]*np.log(p)
                probs[edge] = p**4
            elif mult[edge] == -1:
                weights[edge] = 0
                probs[edge] = 0

        
        nx.set_edge_attributes(rG,weights,"weight")   
        nx.set_edge_attributes(rG,weights,"error_probability")        

        edge_mapping = dict()
        inv_edge_mapping = dict()
        i = 0
        for u,v in rG.edges():
            edge_mapping[i] = (u,v)
            inv_edge_mapping[(u,v)] = i
            i += 1
        nx.set_edge_attributes(rG,inv_edge_mapping,"fault_ids")

        # relabel nodes with integers
        inv_node_mapping = dict([(old_label, i)
                       for i, old_label 
                       in enumerate(rG.nodes())])
        node_mapping = dict([(i,old_label)
                       for i, old_label 
                       in enumerate(rG.nodes())])
        
        rG = nx.relabel_nodes(rG,inv_node_mapping)

        # make matching
        m = pym.Matching(rG)
        return (m,node_mapping,inv_node_mapping,edge_mapping,inv_edge_mapping)
        
    def mark_restricted_edges (self, rG, edges):
        edge_marks = dict()
        for edge in rG.edges:
            if edge in edges:
                edge_marks[edge] = True
            else:
                edge_marks[edge] = False
        nx.set_edge_attributes(rG,edge_marks,"marked")
    
    def unmark_restricted_edges (self, rG):
        edge_marks = dict()
        for edge in rG.edges:
            edge_marks[edge] = False
        nx.set_edge_attributes(rG,edge_marks,"marked")

    def decode_restricted_single (rG,m,node_map, inv_node_map, edge_map, inv_edge_map):
        syndrome = [0]*m.num_nodes
        flipped_checks = [inv_node_map[check] for check,data in rG.nodes(data=True) if data["faulty"]]
        for check in flipped_checks:
            syndrome[check] = 1

        int_decoding = m.decode(syndrome)
        decoding = list()
        for i in range(0,len(int_decoding)):
            if int_decoding[i] == 1:
                decoding.append(edge_map[i])
        
        return decoding
    
    def identify_logical (self, errors, logicals):

        logical_coset = [0] * len(logicals)

        for i in range(0,len(logicals)):
            if bool(len(set(errors).intersection(logicals[i]))%2):
                    logical_coset[i] += 1
        return logical_coset




        

        

        
            


    



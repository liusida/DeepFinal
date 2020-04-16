import copy, math, random, json
import numpy as np
import networkx as nx
from .CPPNActivationFunctions import *

activation_functions = [np.sin, np.abs, neg_abs, np.square, neg_square, sqrt_abs, neg_sqrt_abs]
activation_name_to_fn = {}
for fn in activation_functions:
    activation_name_to_fn[fn.__name__] = fn

"""
for ALife paper, use (0: empty, 1: passiveSoft, 2: passiveHard, 3: active+, 4:active-)
shape: 0 or other
muscleOrTissue: 1/2 or 3/4
tissueType: 1 or 2
muscleType: 3 or 4
phaseoffset: not used.
"""

class CPPN:
    input_node_names = ['x', 'y', 'z', 'd', 'b']
    # do resolution, no need. # output_node_names = ['body', 'phaseoffset','bone_proportion']
    output_node_names = ['body', 'shape', 'muscleOrTissue', 'tissueType', 'muscleType', 'phaseoffset']
    def __init__(self):
        # There are many differences between networkx 1.x and 2.x, we'll use 2.x
        assert float(nx.__version__)>2.0
        self.hidden_node_names = []

    def init(self, hidden_layers, weight_mutation_std):
        self.hidden_layers = hidden_layers
        self.weight_mutation_std = weight_mutation_std
        self.init_graph()

    def clone(self):
        ret = copy.copy(self)
        ret.graph = self.graph.copy()
        return ret

    def __str__(self):
        return self.dumps()
    
    def dumps(self):
        """ Serierize CPPN class. Save all the graph into vxd file, so that we can load from a vxd later. """
        ret = {}
        ret["input_node_names"] = self.input_node_names
        ret["output_node_names"] = self.output_node_names
        ret["hidden_node_names"] = self.hidden_node_names
        ret["hidden_layers"] = self.hidden_layers
        weights = {}
        activation = {}

        for node1,node2 in self.graph.edges:
            weights[f"{node1}__{node2}"] = self.graph.edges[node1, node2]["weight"]
        
        for name in self.hidden_node_names:
            activation[name] = self.graph.nodes[name]["function"].__name__

        ret["weights"] = weights
        ret["activation"] = activation
        return json.dumps(ret)

    def loads(self, s):
        """ Load class from a string, which is probably stored in a vxd. """
        obj = json.loads(s)
        self.input_node_names = obj["input_node_names"]
        self.output_node_names = obj["output_node_names"]
        self.hidden_node_names = obj["hidden_node_names"]
        self.hidden_layers = obj["hidden_layers"]
        self.init_graph()
        for name in obj["activation"]:
            fn = obj["activation"][name]
            self.graph.add_node(name, function=activation_name_to_fn[fn])
        for str_names in obj["weights"]:
            weight = obj["weights"][str_names]
            node1, node2 = str_names.split("__")
            self.graph.add_edge(node1, node2, weight=weight)

    def init_graph(self):
        """Create a simple graph with each input attached to each output"""
        self.graph = nx.DiGraph()

        nodes_this_layer = []
        for name in self.input_node_names:
            self.graph.add_node(name, type="input", function=None)
            nodes_this_layer.append(name)

        for layer_id, layer in enumerate(self.hidden_layers):
            nodes_last_layer = nodes_this_layer
            nodes_this_layer = []
            for node in range(layer):
                name = f"hidden_{layer_id}_{node}"
                self.graph.add_node(name, type="hidden", function=random.choice(activation_functions))
                for last in nodes_last_layer:
                    self.graph.add_edge(last, name, weight=random.random())
                nodes_this_layer.append(name)
                self.hidden_node_names.append(name)

        nodes_last_layer = nodes_this_layer
        nodes_this_layer = []
        for name in self.output_node_names:
            self.graph.add_node(name, type="output", function=sigmoid)
            for last in nodes_last_layer:
                self.graph.add_edge(last, name, weight=random.random())
            nodes_this_layer.append(name)

    def _compute_value(self, node, input_data):
        if self.graph.nodes[node]["evaluated"]:
            return self.graph.nodes[node]["value"]
        if node in input_data:
            return input_data[node]
        predecessors = self.graph.predecessors(node)
        value = 0.0
        for predecessor in predecessors:
            edge = self.graph.get_edge_data(predecessor, node)
            value += self._compute_value(predecessor,input_data) * edge["weight"]
        if self.graph.nodes[node]["function"] is not None:
            value = self.graph.nodes[node]["function"](value)
        self.graph.nodes[node]["value"] = value
        self.graph.nodes[node]["evaluated"] = True
        return value

    def compute(self, input_data):
        """ return a dictionary, key is the output nodes, value is the output value """
        # for node in self.output_node_names:
        for node in self.graph.nodes:
            self.graph.nodes[node]["evaluated"] = False
        ret = {}
        for node in self.output_node_names:
            ret[node] = self._compute_value(node, input_data)
        return ret

    def draw(self):
        import matplotlib.pyplot as plt
        nx.draw_networkx(self.graph, pos=nx.drawing.nx_pydot.graphviz_layout(self.graph, prog='dot'))
        edge_labels_1 = nx.get_edge_attributes(self.graph,'weight')
        for key in edge_labels_1:
            edge_labels_1[key] = round(edge_labels_1[key],2)
        nx.draw_networkx_edge_labels(self.graph, pos=nx.drawing.nx_pydot.graphviz_layout(self.graph, prog='dot'), edge_labels=edge_labels_1, rotate=False)
        plt.show()

    def mutate(self, num_random_activation_functions=1, num_random_weight_changes=5):
        # for _ in range(num_random_activation_functions):
        #     self.change_activation()
        # for _ in range(num_random_weight_changes):
        #     self.change_weight()
        # return
        total = num_random_activation_functions + num_random_weight_changes
        # choose a mutation according to probability
        while True:
            fn = np.random.choice( [
                self.change_activation,
                self.change_weight,
                ], size=1, p=[
                    num_random_activation_functions / total,
                    num_random_weight_changes   / total,
                    ] )
            # print(fn[0])
            success = fn[0]()
            if success:
                break
            print("Retry.")

    def change_activation(self):
        if len(self.hidden_node_names)==0:
            return False
        node = random.choice(self.hidden_node_names)
        success = False
        for i in range(10):
            activation = random.choice(activation_functions)
            if self.graph.nodes[node]["function"] != activation:
                self.graph.nodes[node]["function"]=activation
                success = True
                break
        return success

    def change_weight(self):
        edge = random.choice(list(self.graph.edges))
        self.graph.edges[edge[0], edge[1]]["weight"] = np.random.normal(loc=self.graph.edges[edge[0], edge[1]]["weight"], scale=self.weight_mutation_std)
        return True

    def get_output(self,body_dimension):
        input_x = np.zeros(body_dimension)
        input_y = np.zeros(body_dimension)
        input_z = np.zeros(body_dimension)

        for i in range(body_dimension[0]):
            x = i*2/body_dimension[0] - 1
            for j in range(body_dimension[1]):
                y = j*2/body_dimension[1] - 1
                for k in range(body_dimension[2]):
                    z = k*2/body_dimension[2] - 1
                    input_x[i,j,k] = x
                    input_y[i,j,k] = y
                    input_z[i,j,k] = z

        input_d = np.sqrt(np.power(input_x,2) + np.power(input_y,2)  + np.power(input_z,2) )
        ret = self.compute({'x':input_x,'y':input_y,'z':input_z,'d':input_d,'b':1})
        return ret

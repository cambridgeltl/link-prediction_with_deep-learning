import sys
import csv
import random

"""
Sets up the data for the Link Prediction experiment.
Given the raw data file, it: 1)removes a specified amount of edges for representation induction, 2) creates a specified amount of negative examples from the remaining edges,
3) splits the positive and negative examples into train, dev and test sets as specified 4) writes the relevant train, dev and test files.

"""

def argparser():
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--input-file', help='Input .tsv file')
    ap.add_argument('-n', '--negative-ratio', default=1, help='Ratio of negative to positive examples to create (default 1)')
    
    ap.add_argument('-tf', '--train-filename', default='train.tsv', help='name of file for training data (default train.tsv)')
    ap.add_argument('-df', '--devel-filename', default='devel.tsv', help='name of file for development data (default devel.tsv)')
    ap.add_argument('-tef', '--test-filename', default='test.tsv', help='name of file for testing data (default test.tsv)')
    
    ap.add_argument('-vf', '--vertices-filename', default='vertices.txt', help='name of file containing mapping of all vertex number to name (default vertices.txt)')
    ap.add_argument('-tgf', '--train-graph-filename', default='train_adj_mat.adjlist', help='name of file to store graph for representation induction for training (default train_adj_mat.adjlist)')
    ap.add_argument('-tegf', '--test-graph-filename', default='test_adj_mat.adjlist', help='name of file to store graph for representation induction for testing (default test_adj_mat.adjlist)')
    
    ap.add_argument('-s', '--split', default='70:10:20', help='Train/devel/test split (default 70:10:20)')
    ap.add_argument('-il', '--induction_learning_split', default='50:50', help='Split for inducing representations and learning model (default 50:50)')
    
    ap.add_argument('-v', '--values', default='0:1', help='Values for labels (default 0:1)')
    ap.add_argument('-l', '--labels', default='O:I-LINK', help='Labels for values (default O:I-LINK)')
    
    ap.add_argument('-x', '--indices', default='0:1', help='Index of tsv file where entities can be found (default 0:1)')
    ap.add_argument('-a', '--attributes', default='score', help='Names of link attributes to be read from data (default score)')
    ap.add_argument('-ci', '--col_indices', default='0:1', help='Index of tsv file where information on entities and attributes are (default 0:1)')
    ap.add_argument('-cl', '--col_labels', default='entity1,entity2', help='Labels of the data in the tsv file where entities and attributes are (default 0:1)')
    ap.add_argument('-sc', '--split_criteria', default=None, help='How to split the data for processing of form: split_name1:criteria_name,operator,criteria_value::split_name2:criteria1_name,operator1,criteria1_value|criteria2_name,operator2,criteria2_value (default None)')
    ap.add_argument('-is', '--induction_split_names', default=None, help='Names of splits to be used for induction of form name1:name2 (default None)')
    ap.add_argument('-tr', '--train_split_names', default=None, help='Names of splits to be used as training data of form name1:name2 (default None)')
    ap.add_argument('-dn', '--devel_split_names', default=None, help='Names of splits to be used as development data of form name1:name2 (default None)')
    ap.add_argument('-te', '--test_split_names', default=None, help='Names of splits to be used as testing data of form name1:name2 (default None)')
    
    ap.add_argument('-b', '--balance-classes', default=False, help='Whether or not to balance the amount of examples of included class (default False)')
    ap.add_argument('-c', '--maintain-connection', default=False, help='Whether or not to maintain connectivity in induced representations graph (default False)')
    ap.add_argument('-g', '--graph_format', default='adjlist', help='Format of matrix representation output {adjlist, edgelist, line} (default adjlist)')
    ap.add_argument('-gb', '--graph_bipartite', default=False, help='Process graph as bipartitie or not esp for creating negatives')
    
    ap.add_argument('-sg', '--save-graph', default='True', help='Whether or not to save graph adjacency list in a file (default True)')
    
    return ap

def read_data(input_data_file, attributes, col_labels, col_indices):
    col_indices_vals = col_indices.split(':')
    col_labels_vals = col_labels.split(',')
    assert(len(col_indices_vals) == len(col_labels_vals)), "Lenghts of Labels ({}) and Indices ({}) do not match.".format(col_labels, col_indices)
    indices = {}
    for label, index in zip(col_labels_vals, col_indices_vals):
        indices[label] = index
        
    split_indices = col_indices.split(':')
    assert(len(split_indices) >= 2), "Incorrect length for indices: {}".format(len(split_indices))
    attribute_indices = {}
    for attribute in attributes.split(':'):
        attribute_indices[attribute] = -1
    for num, index in enumerate(split_indices):
        if num == 0:
            entity1_index = int(index)
        elif num == 1:
            entity2_index = int(index)
        else:
            attribute_index_index = num - 2 #2 values of the indices (entity 1 and 2) are not part of the attributes list
            attribute_indices[attributes.split(':')[attribute_index_index]] = int(index)
            
    with open(input_data_file) as tsv:
        entity1_lst = []
        entity2_lst = []
        
        self_referential_edges = 0
        data = {}
        for ind, line in enumerate(csv.reader(tsv, delimiter="\t")): #quoting=csv.QUOTE_NONE - If req to make data work, examine data
            attribute_values = {}
            entity1 = line[entity1_index].replace(' ', '_')
            entity2 = line[entity2_index].replace(' ', '_')
            #Check for attribute values
            for attribute, index in attribute_indices.iteritems():
                attribute_values[attribute] = line[index]
            
            if entity1 == entity2:
                self_referential_edges += 1
            
            key1 = "%s::%s" % (entity1, entity2)
            key2 = "%s::%s" % (entity2, entity1)
            score = attribute_values['score'] if 'score' in attribute_values else None
            if key1 not in data and key2 not in data:
                if score:
                    try:
                        if int(score) == 0: #Remove possible unconnected nodes
                            continue
                    except:
                        continue
                else:
                    if attributes == '':
                        attribute_values = 1
                #Shuffle node order in key
                if random.choice([0,1]) == 1:
                    data[key1] = attribute_values
                else:
                    data[key2] = attribute_values
                entity1_lst.append(entity1)
                entity2_lst.append(entity2)
                
    print("\n%s nodes read. %s edges read." % (len(set(entity1_lst + entity2_lst)), len(data)))
    print("{}/{:,} ({}%) edges were self-referential.".format(self_referential_edges, len(data), (self_referential_edges/float(len(data)))*100 ))
    return data

def _equals(data, criteria, criteria_value):
    edges = []
    for edge, attributes in data.iteritems():
        assert(criteria in attributes), "Criteria {} not in attributes.".format(criteria)
        if int(attributes[criteria]) == criteria_value:
            edges.append(edge)
    return edges

def _greaterthan(data, criteria, criteria_value):
    edges = []
    for edge, attributes in data.iteritems():
        assert(criteria in attributes), "Criteria {} not in attributes.".format(criteria)
        if int(attributes[criteria]) > criteria_value:
            edges.append(edge)
    return edges

def _lessthan(data, criteria, criteria_value):
    edges = []
    for edge, attributes in data.iteritems():
        assert(criteria in attributes), "Criteria {} not in attributes.".format(criteria)
        if int(attributes[criteria]) < criteria_value:
            edges.append(edge)
    return edges

#e.g of split: {'p1': [('date', '=', 1980)]}
def get_splits(data, split_criteria=None):
    induction_edges, train_edges, test_edges = {}, {}, {}
    
    #split data
    split_data = {}
    for name, criteria_lst in split_criteria.iteritems():
        split_data[name] = []
        for ind, criteria in enumerate(criteria_lst):
            valid = []
            if criteria[1] == '=':
                valid += _equals(data, criteria[0], criteria[2])
            elif criteria[1] == '>':
                valid += _greaterthan(data, criteria[0], criteria[2])
            elif criteria[1] == '<':
                valid += _lessthan(data, criteria[0], criteria[2])
            elif criteria[1] == '>=':
                valid += _greaterthan(data, criteria[0], criteria[2]) + _equals(data, criteria[0], criteria[2])
            elif criteria[1] == '<=':
                valid += _lessthan(data, criteria[0], criteria[2]) + _equals(data, criteria[0], criteria[2])
            else:
                print("ERROR: Invalid criteria {} used.".format(criteria[1]))
            
            if ind > 0:
                valid = set(valid).intersection(set(split_data[name]))
                split_data[name] = list(valid)
            else:
                split_data[name] += list(valid)
    
    return split_data
    
def get_induction_learning_edges(data, induction_learning_split, maintain_connection):
    induction_edges, learning_edges = {}, {}
    
    induction_split = int(induction_learning_split.split(':')[0])
    learning_split = int(induction_learning_split.split(':')[1])
    
    keys = data.keys()
    
    if maintain_connection:
        induction_keys = []
        for key in keys:
            entity1 = key.split('::')[0]
            entity2 = key.split('::')[1]
            if entity1 in induction_keys and entity2 in induction_keys:
                #Both already have some connection
                current_induction_ratio = (float(len(induction_edges))/(len(learning_edges) + len(induction_edges))) * 100
                current_learning_ratio = (float(len(learning_edges))/(len(learning_edges) + len(induction_edges))) * 100
                if current_induction_ratio > induction_split and abs(current_induction_ratio - induction_split) > 5:
                    #Induction edges ratio far above specified split place in learning
                    learning_edges[key] = data[key]
                elif current_learning_ratio > learning_split and abs(current_learning_ratio - learning_split) > 5:
                    #learning edges ratio far above specified split place in induction
                    induction_edges[key] = data[key]
                else:
                    #Both already have some connection, and ratio roughly in line with specified split place randomly
                    place = random.randrange(100)
                    if place < induction_split:
                        induction_edges[key] = data[key]
                    else:
                        learning_edges[key] = data[key]
            else:
                induction_edges[key] = data[key]
                induction_keys.append(entity1)
                induction_keys.append(entity2)
    else:
        for key in keys:
            place = random.randrange(100)
            
            if place < induction_split:
                induction_edges[key] = data[key]
            else:
                learning_edges[key] = data[key]
            
    print("\n%s Induction Edges. %s Learning edges." % (len(induction_edges), len(learning_edges)))
    return induction_edges, learning_edges

def create_adjacency_matrix_file(induction_edges, train_edges, test_edges, train_graph_filename, test_graph_filename, graph_format, vertices_filename, save_graph):
    print("Start of create adj matrices function: Training adj-mat edges %s. Testing adj-mat edges: %s" % (len(induction_edges), len(induction_edges) + len(train_edges)))
    train_adj_mat = {} #Only induction edges
    test_adj_mat = {} #Induction and training edges to create embeddings for testing
    vertices = {}
    
    node_index = 0
    #Add induction edges to training adj_mat
    for key, weight in induction_edges.iteritems():
        entity1 = key.split('::')[0]
        entity2 = key.split('::')[1]
        
        if entity1 not in vertices:
            node_index += 1
            ent1_vertex_ind = node_index
            vertices[entity1] = node_index
        else:
            ent1_vertex_ind = vertices[entity1]
            
        if entity2 not in vertices:
            node_index += 1
            ent2_vertex_ind = node_index
            vertices[entity2] = node_index
        else:
            ent2_vertex_ind = vertices[entity2]
            
        
        if ent1_vertex_ind in train_adj_mat:
           train_adj_mat[ent1_vertex_ind].append(str(ent2_vertex_ind))
        else:
            train_adj_mat[ent1_vertex_ind] = [str(ent2_vertex_ind)]
                
        if ent2_vertex_ind in train_adj_mat:
            train_adj_mat[ent2_vertex_ind].append(str(ent1_vertex_ind))
        else:
            train_adj_mat[ent2_vertex_ind] = [str(ent1_vertex_ind)]
        
    print("\n%s induction vertices read." % len(vertices))
    
    #Copy all train adj_mat to test adj_mat
    for key, value in train_adj_mat.iteritems():
        if key not in test_adj_mat:
            test_adj_mat[key] = value
    
    #Add training edges to testing adj_mat
    new_addition = 0
    already_added = 0
    for key, weight in train_edges.iteritems():
        entity1 = key.split('::')[0]
        entity2 = key.split('::')[1]
        
        if entity1 not in vertices:
            node_index += 1
            ent1_vertex_ind = node_index
            vertices[entity1] = node_index
        else:
            ent1_vertex_ind = vertices[entity1]
            
        if entity2 not in vertices:
            node_index += 1
            ent2_vertex_ind = node_index
            vertices[entity2] = node_index
        else:
            ent2_vertex_ind = vertices[entity2]
            
    
        if ent1_vertex_ind in test_adj_mat:
            if str(ent2_vertex_ind) not in test_adj_mat[ent1_vertex_ind]:
                test_adj_mat[ent1_vertex_ind].append(str(ent2_vertex_ind))
        else:
            new_addition += 1
            test_adj_mat[ent1_vertex_ind] = [str(ent2_vertex_ind)]
                
        if ent2_vertex_ind in test_adj_mat:
            if str(ent1_vertex_ind) not in test_adj_mat[ent2_vertex_ind]:
                test_adj_mat[ent2_vertex_ind].append(str(ent1_vertex_ind))
        else:
            new_addition += 1
            test_adj_mat[ent2_vertex_ind] = [str(ent1_vertex_ind)]

    print("\n%s induction and train vertices read." % len(vertices))
    print("\nAfter adding training edges: %s vertices in Train Matrix representation. %s vertices in Test Matrix representation." % (len(train_adj_mat), len(test_adj_mat)))
   
    #Add possible left out nodes to adj_mats (possible if connectivity not required)
    unconnected_nodes = 0
    connected_nodes = {}
    learning_edges = {}
    for edge_dict in [train_edges, test_edges]:
        for key, value in edge_dict.iteritems():
            learning_edges[key] = value
    for key, weight in learning_edges.iteritems():
        entity1 = key.split('::')[0]
        entity2 = key.split('::')[1]
        
        if entity1 not in vertices:
            node_index += 1
            unconnected_nodes += 1
            vertices[entity1] = node_index
            #Must be a totally new entity, add to both matrices
            train_adj_mat[node_index] = []
            test_adj_mat[node_index] = []
        else:
            n_index = vertices[entity1]
            if n_index not in train_adj_mat:
                train_adj_mat[n_index] = []
            if n_index not in test_adj_mat:
                test_adj_mat[n_index] = []
            if entity1 not in connected_nodes:
                connected_nodes[entity1] = 1
            
        if entity2 not in vertices:
            node_index += 1
            unconnected_nodes += 1
            vertices[entity2] = node_index
            #Must be a totally new entity, add to both matrices
            train_adj_mat[node_index] = []
            test_adj_mat[node_index] = []
        else:
            n_index = vertices[entity2]
            if n_index not in train_adj_mat:
                train_adj_mat[n_index] = []
            if n_index not in test_adj_mat:
                test_adj_mat[n_index] = []
            if entity2 not in connected_nodes:
                connected_nodes[entity2] = 1

    print("Ratio unconnected nodes to connected in learning is {:,}:{:,}.".format(unconnected_nodes, len(connected_nodes)))
    print("\n%s total vertices read. %s vertices in Train Matrix representation. %s vertices in Test Matrix representation." % (len(vertices), len(train_adj_mat), len(test_adj_mat)))
    
    train_output = ""
    test_output = ""
    train_output_cnt = 0
    test_output_cnt = 0
    for graph_format in ['adjlist', 'edgelist', 'line', 'sdne']:
        train_output = ""
        test_output = ""
        train_output_cnt = 0
        test_output_cnt = 0
        train_graph_filename = 'train_adj_mat.{}'.format(graph_format)
        test_graph_filename = 'test_adj_mat.{}'.format(graph_format)
        
        if graph_format == 'adjlist':
            for key, value in train_adj_mat.iteritems():
                if len(value) > 0:
                    train_output_cnt += 1
                    train_output += "%s %s\n" % (key, ' '.join(value))
                else:
                    train_output += "%s\n" % (key)
                    
            for key, value in test_adj_mat.iteritems():
                if len(value) > 0:
                    test_output_cnt += 1
                    test_output += "%s %s\n" % (key, ' '.join(value))
                else:
                    test_output += "%s\n" % (key)
            print("Non zero train: %s. Non zero test: %s" % (train_output_cnt, test_output_cnt))
        elif graph_format == 'edgelist':
            for key, value in train_adj_mat.iteritems():
                if len(value) > 0:
                    for node_edge in value: 
                        train_output += "%s %s\n" % (key, node_edge)
                else:
                    train_output += "%s %s\n" % (key, key) #Hack to create an edge for nodes which have no edges as node2vec does not ceate embeddings for them. NEEDED??
                    
            for key, value in test_adj_mat.iteritems():
                if len(value) > 0:
                    for node_edge in value: 
                        test_output += "%s %s\n" % (key, node_edge)
                else:
                    test_output += "%s %s\n" % (key, key) #Hack to create an edge for nodes which have no edges as node2vec does not ceate embeddings for them. NEEDED??
        elif graph_format == 'line':
            #TODO: Add proper scores here
            for key, value in train_adj_mat.iteritems():
                if len(value) > 0:
                    for node_edge in value: 
                        train_output += "%s %s %s\n" % (key, node_edge, 1)
                else:
                    train_output += "%s %s %s\n" % (key, key, 0)
                    
            for key, value in test_adj_mat.iteritems():
                if len(value) > 0:
                    for node_edge in value: 
                        test_output += "%s %s %s\n" % (key, node_edge, 1)
                else:
                    test_output += "%s %s %s\n" % (key, key, 0)
        elif graph_format == 'sdne':
            #Needs node and edge count at top of file and nodeid must begin at 0
            node_cnt = len(train_adj_mat)
            edge_cnt = 0
            for key, value in train_adj_mat.iteritems():
                key = str(int(key) - 1)
                if len(value) > 0:
                    for node_edge in value: 
                        node_edge = str(int(node_edge) - 1)
                        train_output += "%s %s\n" % (key, node_edge)
                        edge_cnt += 1
                else:
                    train_output += "%s %s\n" % (key, key)
                    edge_cnt += 1
            train_output = "%s %s\n" % (node_cnt, edge_cnt) + train_output
            
            node_cnt = len(test_adj_mat)
            edge_cnt = 0
            for key, value in test_adj_mat.iteritems():
                key = str(int(key) - 1)
                if len(value) > 0:
                    for node_edge in value: 
                        node_edge = str(int(node_edge) - 1)
                        test_output += "%s %s\n" % (key, node_edge)
                        edge_cnt += 1
                else:
                    test_output += "%s %s\n" % (key, key)
                    edge_cnt += 1
            test_output = "%s %s\n" % (node_cnt, edge_cnt) + test_output
        
        fil2 = open(train_graph_filename, 'w')
        fil2.write(train_output)
        
        test_graph = open(test_graph_filename, 'w')
        test_graph.write(test_output)
    
    voutput = ""
    fil3 = open(vertices_filename, 'w')
    for vertex, index in vertices.iteritems():
        voutput += "%s %s\n" % (index, vertex)
    fil3.write(voutput)
    
    if save_graph:
        graph_output = ""
        for key, value in test_adj_mat.iteritems():
            if len(value) > 0:
                graph_output += "%s %s\n" % (key, ' '.join(value))
            else:
                graph_output += "%s\n" % (key)
        graph_file = open('graph.adjlist', 'w')
        graph_file.write(graph_output)
        graph_file.close()
    
def create_learning_splits(learning_edges, negative_ratio, balance_classes, train_filename, devel_filename, test_filename, tdt_split, values, labels, separation_fn=None):
    
    keys = learning_edges.keys()
    entity_set = set([k for key in keys for k in key.split('::')])
    print("\n%s vertices in learning." % len(entity_set))
    
    #Create different sets if graph is bipartite. Avoids creating unrealistic (easier) negatives.
    entity1_lst, entity2_lst = None, None
    if separation_fn:
        entity1_lst, entity2_lst = separation_fn(entity_set)
    
    #Create data files
    train = open(train_filename, 'w')
    devel = open(devel_filename, 'w')
    test = open(test_filename, 'w')
    
    header = "node1\tnode2\tlabel\n"
    
    train_output = "%s" % header
    devel_output = "%s" % header
    test_output = "%s" % header
    
    train_max = int(tdt_split.split(':')[0])  
    devel_max = train_max + int(tdt_split.split(':')[1])
    test_max = devel_max + int(tdt_split.split(':')[2])
    
    values = [int(val) for val in values.split(':')]
    labels = labels.split(':')
    assert (len(values) == len(labels)), "%s values and %s labels" % (len(values), len(labels))
    
    #Split edges into train, devel, test
    added_cnts = {}
    train_cnts = {}
    devel_cnts = {}
    test_cnts = {}
    train_entities = []
    test_entities = []
    train_edges = {}
    devel_edges = {}
    test_edges = {}
    for key, value in learning_edges.iteritems():
        if balance_classes:
            if value in added_cnts:
                if added_cnts[value] <= least_cnts:
                    added_cnts[value] += 1
            else:
                added_cnts[value] = 1
            
            if added_cnts[value] > least_cnts:
                continue
        
        entity1 = key.split('::')[0]
        entity2 = key.split('::')[1]
        
        place = random.randrange(100)
        if place < train_max:
            train_edges[key] = 1
            if value in train_cnts:
                train_cnts[value] += 1
            else:
                train_cnts[value] = 1
            train_entities.append(entity1)
            train_entities.append(entity2)
        elif place < devel_max:
            devel_edges[key] = 1
            if value in devel_cnts:
                devel_cnts[value] += 1
            else:
                devel_cnts[value] = 1
        elif place < test_max:
            test_edges[key] = 1
            if value in test_cnts:
                test_cnts[value] += 1
            else:
                test_cnts[value] = 1
            test_entities.append(entity1)
            test_entities.append(entity2)
        else:
            print("WARNING: Item not placed in train, devel or test due to place index of: %s" % place)
            
    #Create train examples
    pos_cnt = len(train_edges)
    entity_lst = list(entity_set)
    noise_edges = 0
    while len(train_edges) < pos_cnt * (int(negative_ratio) + 1):
        ent1 = random.choice(entity_lst) if not entity1_lst else random.choice(entity1_lst)
        ent2 = random.choice(entity_lst) if not entity2_lst else random.choice(entity2_lst)
        if ent1 != ent2:
            key1 = "%s::%s" % (ent1, ent2)
            key2 = "%s::%s" % (ent2, ent1)
            if key1 not in train_edges and key2 not in train_edges:
                #Shuffle node order in key
                if random.choice([0,1]) == 1:
                    train_edges[key1] = 0
                    #Track whether it is a 'noise edge'
                    if key1 in devel_edges or key1 in test_edges:
                        noise_edges += 1
                else:
                    train_edges[key2] = 0
                    #Track whether it is a 'noise edge'
                    if key2 in devel_edges or key2 in test_edges:
                        noise_edges += 1
    print("There were {}/{:,} ({}%) noise edges in training data.".format(noise_edges, len(train_edges), (noise_edges/float(len(train_edges)))*100 if len(train_edges) > 0 else 0))
                        
    #Create devel examples
    pos_cnt = len(devel_edges)
    noise_edges = 0
    while len(devel_edges) < pos_cnt * (int(negative_ratio) + 1):
        ent1 = random.choice(entity_lst) if not entity1_lst else random.choice(entity1_lst)
        ent2 = random.choice(entity_lst) if not entity2_lst else random.choice(entity2_lst)
        if ent1 != ent2:
            key1 = "%s::%s" % (ent1, ent2)
            key2 = "%s::%s" % (ent2, ent1)
            if key1 not in train_edges and key1 not in devel_edges and key2 not in train_edges and key2 not in devel_edges:
                #Shuffle node order in key
                if random.choice([0,1]) == 1:
                    devel_edges[key1] = 0
                    #Track whether it is a 'noise edge'
                    if key1 in test_edges:
                        noise_edges += 1
                else:
                    devel_edges[key2] = 0
                    #Track whether it is a 'noise edge'
                    if key2 in test_edges:
                        noise_edges += 1
    print("There were {}/{:,} ({}%) noise edges in devel data.".format(noise_edges, len(devel_edges), (noise_edges/float(len(devel_edges)))*100 if len(devel_edges) > 0 else 0))
                        
    #Create test examples
    pos_cnt = len(test_edges)
    noise_edges = 0
    while len(test_edges) < pos_cnt * (int(negative_ratio) + 1):
        ent1 = random.choice(entity_lst) if not entity1_lst else random.choice(entity1_lst)
        ent2 = random.choice(entity_lst) if not entity2_lst else random.choice(entity2_lst)
        if ent1 != ent2:
            key1 = "%s::%s" % (ent1, ent2)
            key2 = "%s::%s" % (ent2, ent1)
            if key1 not in train_edges and key1 not in devel_edges and key1 not in test_edges \
                and key2 not in train_edges and key2 not in devel_edges and key2 not in test_edges:
                #Shuffle node order in key
                if random.choice([0,1]) == 1:
                    test_edges[key1] = 0
                else:
                    test_edges[key2] = 0
                        
    train_cnts = {}
    train_entities = []
    for key, value in train_edges.iteritems():
        entity1 = key.split('::')[0]
        entity2 = key.split('::')[1]
        
        assert (value in values), "Value %s not in values." % value
        label = labels[values.index(value)]
        entry = "%s\t%s\t%s\n" % (entity1, entity2, label)
        
        train_output += entry
        if value in train_cnts:
            train_cnts[value] += 1
        else:
            train_cnts[value] = 1
        train_entities.append(entity1)
        train_entities.append(entity2)
        
    devel_cnts = {}
    devel_entities = []
    for key, value in devel_edges.iteritems():
        entity1 = key.split('::')[0]
        entity2 = key.split('::')[1]
        
        assert (value in values), "Value %s not in values." % value
        label = labels[values.index(value)]
        entry = "%s\t%s\t%s\n" % (entity1, entity2, label)
        
        devel_output += entry
        if value in devel_cnts:
            devel_cnts[value] += 1
        else:
            devel_cnts[value] = 1
        devel_entities.append(entity1)
        devel_entities.append(entity2)
    
    test_cnts = {}
    test_entities = []
    print("There are %s keys in test edges" % len(test_edges))
    for key, value in test_edges.iteritems():
        entity1 = key.split('::')[0]
        entity2 = key.split('::')[1]
        
        assert (value in values), "Value %s not in values." % value
        label = labels[values.index(value)]
        entry = "%s\t%s\t%s\n" % (entity1, entity2, label)
        
        test_output += entry
        if value in test_cnts:
            test_cnts[value] += 1
        else:
            test_cnts[value] = 1
        test_entities.append(entity1)
        test_entities.append(entity2)
    
    train_set = set(train_entities)
    test_set = set(test_entities)
    train_set_size = len(train_set)
    test_set_size = len(test_set)
    train_test_intersection_size = len(train_set.intersection(test_set))
                                       
    print("\nEntities in train only: {:,}".format(train_set_size - train_test_intersection_size))
    print("Entities in test only: {:,}".format(test_set_size - train_test_intersection_size))
    print("Entities in both train and test: {:,}".format(train_test_intersection_size))
    
    print("\nClass items counts:")
    print("Train counts:")
    for k, v in train_cnts.iteritems():
        print("{}\t: {:,}".format(k,v))
    print("Devel counts:")
    for k, v in devel_cnts.iteritems():
        print("{}\t: {:,}".format(k,v))
    print("Test counts:")
    for k, v in test_cnts.iteritems():
        print("{}\t: {:,}".format(k,v))
        
    train.write(train_output)
    devel.write(devel_output)
    test.write(test_output)
    
    #Get only positive train edges
    pos_train = {}
    for edge, score in train_edges.iteritems():
        if score > 0:
            pos_train[edge] = score
    pos_test = {}
    for edge, score in test_edges.iteritems():
        if score > 0:
            pos_test[edge] = score
    print("End of creating learning splits. Train edges count: %s. Test edges count: %s" % (len(pos_train), len(pos_test))) 
    return pos_train, pos_test

    
def create_learning_files(induction_edges, train_edges, devel_edges, test_edges, negative_ratio, train_filename, devel_filename, test_filename, values, labels, fold_values=True, separation_fn=None):
    
    keys = induction_edges.keys() + train_edges.keys() + devel_edges.keys() + test_edges.keys()
    entity_set = set([k for key in keys for k in key.split('::')])
    print("\n%s vertices in learning." % len(entity_set))
    
    entity1_lst, entity2_lst = None, None
    if separation_fn:
        entity1_lst, entity2_lst = separation_fn(entity_set)
    
    if fold_values:
        for edge_set in [train_edges, devel_edges, test_edges]:
            for key, value in edge_set.iteritems():
                if value > 0:
                    edge_set[key] = 1
                else:
                    edge_set[key] = 0
        
    #Create data files
    train = open(train_filename, 'w')
    devel = open(devel_filename, 'w')
    test = open(test_filename, 'w')
    
    header = "node1\tnode2\tlabel\n"
    train_output = "%s" % header
    devel_output = "%s" % header
    test_output = "%s" % header
    
    values = [int(val) for val in values.split(':')]
    labels = labels.split(':')
    assert (len(values) == len(labels)), "%s values and %s labels" % (len(values), len(labels))
    
    #Create train examples
    pos_cnt = len(train_edges)
    entity_lst = list(entity_set)
    noise_edges = 0
    while len(train_edges) < pos_cnt * (int(negative_ratio) + 1):
        ent1 = random.choice(entity_lst) if not entity1_lst else random.choice(entity1_lst)
        ent2 = random.choice(entity_lst) if not entity2_lst else random.choice(entity2_lst)
        if ent1 != ent2:
            key1 = "%s::%s" % (ent1, ent2)
            key2 = "%s::%s" % (ent2, ent1)
            if key1 not in induction_edges and key1 not in train_edges and key2 not in induction_edges and key2 not in train_edges:
                #Shuffle node order in key
                if random.choice([0,1]) == 1:
                    train_edges[key1] = 0
                    #Track whether it is a 'noise edge'
                    if key1 in devel_edges or key1 in test_edges:
                        noise_edges += 1
                else:
                    train_edges[key2] = 0
                    #Track whether it is a 'noise edge'
                    if key2 in devel_edges or key2 in test_edges:
                        noise_edges += 1
    print("There were {}/{:,} ({}%) noise edges in training data.".format(noise_edges, len(train_edges), (noise_edges/float(len(train_edges)))*100 if len(train_edges) > 0 else 0))
                        
    #Create devel examples
    pos_cnt = len(devel_edges)
    noise_edges = 0
    while len(devel_edges) < pos_cnt * (int(negative_ratio) + 1):
        ent1 = random.choice(entity_lst) if not entity1_lst else random.choice(entity1_lst)
        ent2 = random.choice(entity_lst) if not entity2_lst else random.choice(entity2_lst)
        if ent1 != ent2:
            key1 = "%s::%s" % (ent1, ent2)
            key2 = "%s::%s" % (ent2, ent1)
            if key1 not in induction_edges and key1 not in train_edges and key1 not in devel_edges and key2 not in induction_edges and key2 not in train_edges and key2 not in devel_edges:
                #Shuffle node order in key
                if random.choice([0,1]) == 1:
                    devel_edges[key1] = 0
                    #Track whether it is a 'noise edge'
                    if key1 in test_edges:
                        noise_edges += 1
                else:
                    devel_edges[key2] = 0
                    #Track whether it is a 'noise edge'
                    if key2 in test_edges:
                        noise_edges += 1
    print("There were {}/{:,} ({}%) noise edges in devel data.".format(noise_edges, len(devel_edges), (noise_edges/float(len(devel_edges)))*100 if len(devel_edges) > 0 else 0))
                        
    #Create test examples
    pos_cnt = len(test_edges)
    noise_edges = 0
    while len(test_edges) < pos_cnt * (int(negative_ratio) + 1):
        ent1 = random.choice(entity_lst) if not entity1_lst else random.choice(entity1_lst)
        ent2 = random.choice(entity_lst) if not entity2_lst else random.choice(entity2_lst)
        if ent1 != ent2:
            key1 = "%s::%s" % (ent1, ent2)
            key2 = "%s::%s" % (ent2, ent1)
            if key1 not in induction_edges and key1 not in train_edges and key1 not in devel_edges and key1 not in test_edges \
                and key2 not in induction_edges and key2 not in train_edges and key2 not in devel_edges and key2 not in test_edges:
                #Shuffle node order in key
                if random.choice([0,1]) == 1:
                    test_edges[key1] = 0
                else:
                    test_edges[key2] = 0
                        
    train_cnts = {}
    train_entities = []
    for key, value in train_edges.iteritems():
        entity1 = key.split('::')[0]
        entity2 = key.split('::')[1]
        
        assert (value in values), "Value %s not in values." % value
        label = labels[values.index(value)]
        entry = "%s\t%s\t%s\n" % (entity1, entity2, label)
        
        train_output += entry
        if value in train_cnts:
            train_cnts[value] += 1
        else:
            train_cnts[value] = 1
        train_entities.append(entity1)
        train_entities.append(entity2)
        
    devel_cnts = {}
    devel_entities = []
    for key, value in devel_edges.iteritems():
        entity1 = key.split('::')[0]
        entity2 = key.split('::')[1]
        
        assert (value in values), "Value %s not in values." % value
        label = labels[values.index(value)]
        entry = "%s\t%s\t%s\n" % (entity1, entity2, label)
        
        devel_output += entry
        if value in devel_cnts:
            devel_cnts[value] += 1
        else:
            devel_cnts[value] = 1
        devel_entities.append(entity1)
        devel_entities.append(entity2)
    
    test_cnts = {}
    test_entities = []
    for key, value in test_edges.iteritems():
        entity1 = key.split('::')[0]
        entity2 = key.split('::')[1]
        
        assert (value in values), "Value %s not in values." % value
        label = labels[values.index(value)]
        entry = "%s\t%s\t%s\n" % (entity1, entity2, label)
        
        test_output += entry
        if value in test_cnts:
            test_cnts[value] += 1
        else:
            test_cnts[value] = 1
        test_entities.append(entity1)
        test_entities.append(entity2)
    
    train_set = set(train_entities)
    test_set = set(test_entities)
    train_set_size = len(train_set)
    test_set_size = len(test_set)
    train_test_intersection_size = len(train_set.intersection(test_set))
                                       
    print("\nEntities in train only: {:,}".format(train_set_size - train_test_intersection_size))
    print("Entities in test only: {:,}".format(test_set_size - train_test_intersection_size))
    print("Entities in both train and test: {:,}".format(train_test_intersection_size))
    
    print("\nClass items counts:")
    print("Train counts:")
    for k, v in train_cnts.iteritems():
        print("{}\t: {:,}".format(k,v))
    print("Devel counts:")
    for k, v in devel_cnts.iteritems():
        print("{}\t: {:,}".format(k,v))
    print("Test counts:")
    for k, v in test_cnts.iteritems():
        print("{}\t: {:,}".format(k,v))
        
    train.write(train_output)
    devel.write(devel_output)
    test.write(test_output)
    
    return train_edges, test_edges

#In MATADOR dataset, all Chemical identifiers can be integers but no proteins can be
def matador_separation(entities):
    proteins = []
    chemicals = []
    for entity_name in entities:
        try:
            int(entity_name)
            chemicals.append(entity_name)
        except:
            proteins.append(entity_name)
            
    print("For MATADOR dataset: %s proteins and %s chemicals." % (len(proteins), len(chemicals)))
    return proteins, chemicals

def setup_experiment(input_data_file,  attributes, col_labels, col_indices, split_criteria, induction_split_names, train_split_names, devel_split_names, test_split_names, train_graph_filename, test_graph_filename, graph_format,
                     vertices_filename, train_filename, devel_filename, test_filename, negative_ratio, values, labels, induction_learning_split, maintain_connection, balance_classes, tdt_split, save_graph, separation_fn=None): 
    
    #split_name1:criteria_name,operator,criteria_value::split_name2:criteria_name,operator,criteria_value
    #Prepare split criteria data structure
    split_criteria_dict = None
    if split_criteria:
        split_criteria_dict = {}
        criteria_lst = split_criteria.split('::')
        for criteria in criteria_lst:
            split = criteria.split(':')
            assert(len(split) == 2), "Criteria split into {} is invalid.".format(len(split))
            split_name = split[0]
            split_details_lst = split[1].split('|')
            split_criteria_dict[split_name] = []
            for detail in split_details_lst:
                split_details = detail.split(',')
                assert(len(split_details) == 3), "Criteria details split into {} is invalid.".format(len(split_details))
                criteria_name = split_details[0]
                operator = split_details[1]
                criteria_value = split_details[2]
                
                split_criteria_dict[split_name].append((criteria_name, operator, int(criteria_value)))
                
    
    data = read_data(input_data_file, attributes, col_labels, col_indices)
    if split_criteria:
        print("Splitting by criteria.")
        attribute = attributes.split(':')
        if len(attribute) > 1:
            raise NotImplemented('Multiple Attributes not implemented.')
        else:
            attribute = attribute[0]
        split_data = get_splits(data, split_criteria_dict)
        
        assert(induction_split_names and train_split_names and test_split_names), "induction_split_names and train_split_names and devel_split_names and test_split_names must be set if split_criteria set."
        induction_edges = {}
        learning_edges = {}
        train_edges = {}
        devel_edges = {}
        test_edges = {}
        induction_split_names = induction_split_names.split(':')
        train_split_names = train_split_names.split(':')
        devel_split_names = devel_split_names.split(':') if devel_split_names else []
        test_split_names = test_split_names.split(':')
        for name in induction_split_names:
            if name not in split_data:
                print("ERROR: Name {} not in data (Split Names: {}).".format(name, split_data.keys()))
                break
            for key in split_data[name]:
                induction_edges[key] = data[key][attribute]
        print("{:,} induction edges.".format(len(induction_edges)))
        for name in train_split_names:
            if name not in split_data:
                print("ERROR: Name {} not in data (Split Names: {}).".format(name, split_data.keys()))
                break
            for key in split_data[name]:
                train_edges[key] = data[key][attribute]
        for name in devel_split_names:
            if name not in split_data:
                print("ERROR: Name {} not in data (Split Names: {}).".format(name, split_data.keys()))
                break
            for key in split_data[name]:
                devel_edges[key] = data[key][attribute]
        for name in test_split_names:
            if name not in split_data:
                print("ERROR: Name {} not in data (Split Names: {}).".format(name, split_data.keys()))
                break
            for key in split_data[name]:
                test_edges[key] = data[key][attribute]
        
        for edge_set in [train_edges, devel_edges, test_edges]:
            for key in edge_set:
                learning_edges[key] = data[key][attribute]
            
        create_adjacency_matrix_file(induction_edges, train_edges, test_edges, train_graph_filename, test_graph_filename, graph_format, vertices_filename, save_graph)
        create_learning_files(induction_edges, train_edges, devel_edges, test_edges, negative_ratio, train_filename, devel_filename, test_filename, values, labels, separation_fn=separation_fn)
    else:
        print("Splitting by size.")
        induction_edges, learning_edges = get_induction_learning_edges(data, induction_learning_split, maintain_connection)
        train_edges, test_edges = create_learning_splits(learning_edges, negative_ratio, balance_classes, train_filename, devel_filename, test_filename, tdt_split, values, labels, separation_fn=separation_fn)
        create_adjacency_matrix_file(induction_edges, train_edges, test_edges, train_graph_filename, test_graph_filename, graph_format, vertices_filename, save_graph)

def main(argv):
    args = argparser().parse_args(argv[1:])
    if args.graph_bipartite:
         setup_experiment(args.input_file, args.attributes, args.col_labels, args.col_indices, args.split_criteria, args.induction_split_names, args.train_split_names, args.devel_split_names, args.test_split_names, args.train_graph_filename,
                     args.test_graph_filename, args.graph_format, args.vertices_filename, args.train_filename, args.devel_filename, args.test_filename, args.negative_ratio, args.values, args.labels, args.induction_learning_split, args.maintain_connection, args.balance_classes, args.split, args.save_graph, matador_separation) 
    else:
        setup_experiment(args.input_file, args.attributes, args.col_labels, args.col_indices, args.split_criteria, args.induction_split_names, args.train_split_names, args.devel_split_names, args.test_split_names, args.train_graph_filename,args.test_graph_filename, args.graph_format, args.vertices_filename, args.train_filename, args.devel_filename, args.test_filename, args.negative_ratio, args.values, args.labels, args.induction_learning_split, args.maintain_connection, args.balance_classes, args.split, args.save_graph)
            

if __name__ == '__main__':
    sys.exit(main(sys.argv))
    
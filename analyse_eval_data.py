import csv

"""
This file does the analysis of true positives with no prior common neighbours as reported in Section 6.4 of the paper.
"""

def get_positives(input_data_file = 'test.tsv'):
    with open(input_data_file) as tsv:
        positives = []
        for ind, line in enumerate(csv.reader(tsv, delimiter="\t")): #quoting=csv.QUOTE_NONE - If req to make data work, examine data
            if line[2] == 'I-LINK':
                entity1 = line[0].replace(' ', '_')
                entity2 = line[1].replace(' ', '_')
                positives.append("{}::{}".format(entity1, entity2))
                
    return positives
        
def get_graph(vertices_data_file = 'vertices.txt', graph_data_file = 'graph.adjlist'):
    #Read in vertices and node indexes
    vertices = {}
    vertices_file = open(vertices_data_file, 'r')
    for line in vertices_file:
        line = line.split()
        vertices[line[0]] = line[1]

    #Read in graph adjacency list
    graph = {}
    graph_file = open(graph_data_file, 'r')
    for line in graph_file:
        line = line.split()
        if line[0] in vertices:
            node = vertices[line[0]]
            graph[node] = [vertices[node_] for node_ in line[1:]]
        else:
            print("Node index: %s not in vertices." % line[0])
    return graph
            

def calc_recall_limit(graph, test_positives, bipartite, cn_threshold=0):
    
    limited = 0
    for edge in test_positives:
        entity1 = edge.split('::')[0]
        entity2 = edge.split('::')[1]
        
        cn = get_common_neighbours(graph, entity1, entity2, bipartite)
        if int(cn) <= int(cn_threshold):
            limited += 1
            
    print("{}/{} positives less than {}.".format(limited, len(test_positives), cn_threshold))
    return limited/float(len(test_positives))


def get_common_neighbours(graph, entity1, entity2, bipartite):
    entity1_set = set(graph[entity1])
    entity2_set = set(graph[entity2])
        
    if bipartite:
        #Get neighbours of neighbours of this node to use
        entity2_lst = []
        for node in entity2_set:
            entity2_lst += graph[node]
        entity2_set = set(entity2_lst)
        
    cn = len(entity2_set.intersection(entity1_set))
    
    return cn

def get_recall_limited_edges(graph, test_positives, bipartite, cn_threshold=0):
    
    limited = []
    for edge in test_positives:
        entity1 = edge.split('::')[0]
        entity2 = edge.split('::')[1]
        
        cn = get_common_neighbours(graph, entity1, entity2, bipartite)
        if int(cn) <= int(cn_threshold):
            limited.append(edge)

    return limited

def get_overall_rankings(edge_lst, ranked_filename):
    print("Analysing: {}".format(ranked_filename))
    with open(ranked_filename) as tsv:
        ranked_edges = {}
        for ind, line in enumerate(csv.reader(tsv, delimiter="\t")):
            if ind == 0:
                continue
            ranked_edges[line[0]] = ind
            
    rankings = []
    first_quartile = []
    second_quartile = []
    third_quartile = []
    fourth_quartile = []
    total_edges = len(ranked_edges)
    quartile_size = total_edges / 4
    
    for edge in edge_lst:
        rank = ranked_edges[edge]
        if rank < quartile_size:
            quartile = 'First'
            first_quartile.append(edge)
        elif rank < (quartile_size * 2):
            quartile = 'Second'
            second_quartile.append(edge)
        elif rank < (quartile_size * 3):
            quartile = 'Third'
            third_quartile.append(edge)
        else:
            quartile = 'Fourth'
            fourth_quartile.append(edge)
        
        rankings.append((edge, "{} ({})".format(quartile, rank)))
    print("Total: {}".format(total_edges))
    print("First: {}% ({}/{}). Second: {}% ({}/{}). Third: {}% ({}/{}). Fourth: {}% ({}/{}).".format((len(first_quartile)/float(len(edge_lst))) * 100, len(first_quartile), len(edge_lst),
                                                                                                     (len(second_quartile)/float(len(edge_lst))) * 100, len(second_quartile), len(edge_lst),
                                                                                                     (len(third_quartile)/float(len(edge_lst))) * 100, len(third_quartile), len(edge_lst),
                                                                                                     (len(fourth_quartile)/float(len(edge_lst))) * 100, len(fourth_quartile), len(edge_lst)))
    
            
def main():
    bipartite = False
    cn_threshold = 0
    
    #Get these logs by uncommenting line in link_prediction.py calling log_predictions()
    folder = '/path/to/logs'
    
    print("Getting positives...")
    positives = get_positives('{}/test.tsv'.format(folder))
    
    print("Reading graph...")
    graph = get_graph('{}/vertices.txt'.format(folder), '{}/graph.adjlist'.format(folder))
    
    print("Calculating recall limitedness...")
    recall_limit = calc_recall_limit(graph, positives, bipartite, cn_threshold)
    
    print("{}% of the positives had no common neighbours.".format(recall_limit * 100))
    print("Calculating rankings...")
    recall_limited_edges = get_recall_limited_edges(graph, positives, bipartite, cn_threshold)
    get_overall_rankings(recall_limited_edges, '{}/ranked-edges-all-common_neighbours.tsv'.format(folder))
    get_overall_rankings(recall_limited_edges, '{}/ranked-edges-all-deepwalk-concatenate.tsv'.format(folder))
    
main()
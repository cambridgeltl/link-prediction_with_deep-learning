import sys
import csv
import random

"""
The representation creation algorithms may require the node labels to be integers (DeepWalk does). Thus all nodes are mapped to integers.
This code primarily takes the embedidngs which are produced and change them back to the original labels which are strings which may or may not be convertable to integers. 
"""


def argparser():
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--embeddings-filename', help='Embeddings file for input')
    
    ap.add_argument('-vf', '--vertices-filename', default='vertices.txt', help='name of file containing mapping of all vertex number to name (default vertices.txt)')
    ap.add_argument('-o', '--modified-embeddings-filename', default='modified_embeddings.embeddings', 
                    help='name of file to output modified embeddings to (default modified_embeddings.embeddings)')
    
    ap.add_argument('-tgf', '--train-data-filename', default='train-graph.tsv', help='name of file with data to create training graph (default train-graph.tsv)')
    ap.add_argument('-dgf', '--devel-data-filename', default='devel-graph.tsv', help='name of file with data to create development graph (default devel-graph.tsv)')
    ap.add_argument('-tegf', '--test-data-filename', default='test-graph.tsv', help='name of file with data to create testing graph (default test-graph.tsv)')
    
    return ap


def create_modified_embeddings(embeddings_filename, vertices_filename, modified_embeddings_filename):
    #Create modified embedding vectors file

    #Read vertices mapping
    vertices_dict = {}
    with open(vertices_filename) as vertices_file:
        line = vertices_file.readline()
        while line:
            line = line.split()
            vertices_dict[line[1]] = line[0]
            line = vertices_file.readline()
    print("Vertices count: %s" % len(vertices_dict))

    #Read embeddings
    embeddings_dict = {}
    with open(embeddings_filename) as embeddings_file:
        line = embeddings_file.readline() #Read header
        line = embeddings_file.readline()
        while line:
            line = line.split()
            embeddings_dict[line[0]] = line[1:]
            line = embeddings_file.readline()
    print("Embeddings count: %s" % len(embeddings_dict))

    modified_embeddings = open(modified_embeddings_filename, 'w')
    modified_embeddings_output = ""
    negatives_in = 0
    
    already_added = []
    for key,value in vertices_dict.iteritems():
        entity1 = key
        if (entity1 not in already_added) and (entity1 in vertices_dict and vertices_dict[entity1] in embeddings_dict):
            vecs = embeddings_dict[vertices_dict[entity1]]
            vecs_str = " ".join([str(vec) for vec in vecs])
            modified_embeddings_output += "%s %s\n" % (entity1, vecs_str)
            already_added.append(entity1)
        else:
            if entity1 not in vertices_dict:
                print("Entity %s not in vertices dict" % entity1) 
            else:
                if vertices_dict[entity1] not in embeddings_dict:
                    print("Entity %s not in embeddings dict" % entity1)
 
    print("%s embeddings in modified embeddings." % len(already_added))
    
    modified_embeddings.write(modified_embeddings_output)

def main(argv):
    args = argparser().parse_args(argv[1:])

    create_modified_embeddings(args.embeddings_filename, args.vertices_filename, args.modified_embeddings_filename)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

        
        
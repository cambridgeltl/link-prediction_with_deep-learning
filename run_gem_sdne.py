
import sys
from gem.embedding.sdne import SDNE as sdne
from gem.utils import graph_util
import numpy as np

def argparser():
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--graph-filename', help='File with edges for graph input')
    ap.add_argument('-o', '--modified-filename', default='converted_mat.txt', help='name of output file (default converted_mat.txt)')
    ap.add_argument('-vf', '--vertices-filename', default='vertices.txt', help='name of file containing mapping of all vertex number to name (default vertices.txt)')
    
    return ap

def run_sdne(edges_file, modified_filename, vertices_filename):
    # Instatiate the embedding method with hyperparameters
    em = sdne(d=100, beta=5, alpha=1e-6, nu1=1e-3, nu2=1e-3, K=3, n_units=[500, 300], rho=0.3, n_iter=1, xeta=1e-4, n_batch=500, modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'], weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5'])

    # Load graph
    graph = graph_util.loadGraphFromEdgeListTxt(edges_file)

    # Learn embedding - accepts a networkx graph or file with edge list
    Y, t = em.learn_embedding(graph, edge_f=None, is_weighted=True, no_python=True)

    create_converted_file(em.get_embedding(None), modified_filename, vertices_filename)
    
def create_converted_file(embeddings, modified_filename, vertices_filename):
    #Create modified embedding vectors file

    converted_file = open(modified_filename, 'w')
    vertices_file = open(vertices_filename, 'r')
    output = ''
    for line, embedding in zip(vertices_file, embeddings):
        line = line.split()
        if len(line) == 2:
            entity = line[1]
            output += "%s %s\n" % (entity, ' '.join([str(round(e,13)) for e in embedding])) #[str(round(e,15)) for e in embedding]
        else:
            print("Error: Line %s in vertices did not have 2 items." % line) 
    converted_file.write(output)

def main(argv):
    args = argparser().parse_args(argv[1:])

    run_sdne(args.graph_filename, args.modified_filename, args.vertices_filename)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
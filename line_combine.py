import sys
import random
from sklearn import preprocessing

def argparser():
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-i1', '--order1-input-filename', help='Input matrix file from LINE order 1')
    ap.add_argument('-i2', '--order2-input-filename', help='Input matrix file from LINE order 2')
    ap.add_argument('-o', '--embeddings-filename', help='Embeddings file to output')
    
    return ap
    
def combine_vectors(order1_input_file, order2_input_file, output_file):
    
    o1_in_file = open(order1_input_file, 'r')
    o2_in_file = open(order2_input_file, 'r')
    o1_line = o1_in_file.readline()
    o2_line = o2_in_file.readline()
    
    vectors = []
    keys = []
    
    while o1_line and o2_line:
        o1_line = o1_line.split()
        o2_line = o2_line.split()
        assert(o1_line[0] == o2_line[0]), "%s and %s are not the same." % (o1_line[0], o2_line[0])
        if len(o1_line) == len(o2_line) and len(o1_line) == 2:
            print("WARNING: Skipping a line because it appears to be header line.")
            o1_line = o1_in_file.readline()
            o2_line = o2_in_file.readline()
            continue
        vector = [val for val in o1_line[1:]] + [val for val in o2_line[1:]]
        vectors.append(vector)
        keys.append(o1_line[0])
        o1_line = o1_in_file.readline()
        o2_line = o2_in_file.readline()
        
    vector_length = len(vectors[0])
    vector_cnt = len(vectors)
    vectors = preprocessing.normalize(vectors)
    output = ""
    for key, vector in zip(keys, vectors):
        output += "%s %s\n" % (key, ' '.join([str(num) for num in vector]))
    out_file = open(output_file, 'w')
    output = "%s %s\n%s" % (vector_cnt, vector_length, output)
    out_file.write(output)
    
def main(argv):
    args = argparser().parse_args(argv[1:])

    combine_vectors(args.order1_input_filename, args.order2_input_filename, args.embeddings_filename)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

import argparse
import sys
import tempfile
import math

import random
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import estimator
import tflearn
import numpy as np
from sklearn import metrics
from collections import Counter
from datetime import datetime

import wordvecdata as wvd

"""
Implements neural classifier in tensoflow. Also implements the metrics used in the work.
Disclaimer: File probably needs tons of cleaning up...
"""

COLUMNS = ["node1", "node2"]
LABEL_COLUMN = "label"

def build_estimator(model_dir, model_type, embeddings,index_map, combination_method):
  """Build an estimator."""

  # Continuous base columns.
  node1 = tf.contrib.layers.real_valued_column("node1")

  deep_columns = [node1]
  
  if model_type == "regressor":
      
      tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)
      if combination_method == 'concatenate':
          net = tflearn.input_data(shape=[None, embeddings.shape[1]*2])
      else:
        net = tflearn.input_data(shape=[None, embeddings.shape[1]] )
      net = tflearn.fully_connected(net, 100, activation='relu')
      net = tflearn.fully_connected(net, 2, activation='softmax')
      net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
      m = tflearn.DNN(net)
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100])
  return m

def get_input(df, embeddings,index_map, combination_method='hadamard', data_purpose='train'):
  """Input builder function."""
  # Converts the label column into a constant Tensor.
  label_values = df[LABEL_COLUMN].values
  
  indexed_labels = []
  original_labels = np.array(label_values)
  labels = [[0, 0] for i in range(len(label_values))]
  
  for label_lst, value in zip(labels, label_values):
      label_lst[value] = 1
  indexed_labels = labels
      
  if data_purpose not in ['map', 'test']:
    vocab_size = embeddings.shape[0]
    embedding_dim = embeddings.shape[1]
    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="W")

    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embeddings)
    
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embeddings)
    
  feature_cols = {}
  column_tensors = []
  col_keys = []
  for i in COLUMNS:
    words = [value for value in df[i].values]
    col_keys.append(words)
    #print("%s words in index map." % len(index_map))
    ids = [index_map[word] for word in words]
    column_tensors.append([embeddings[id_] for id_ in ids])
    
  keys = []
  for entity1, entity2 in zip(col_keys[0], col_keys[1]):
      keys.append("%s::%s" % (entity1, entity2)) 
    
  assert(combination_method in ['hadamard','average', 'weighted_l1', 'weighted_l2', 'concatenate']), "Invalid combination Method %s" % combination_method
  
  features = column_tensors[0]
  no_output = ['map']
  for i in range(1, len(column_tensors)):
      if combination_method == 'hadamard':
          if data_purpose not in no_output:
            print("Combining with Hadamard.")
          features = np.multiply(features, column_tensors[i])
      elif combination_method == 'average':
          if data_purpose not in no_output:
            print("Combining with Average.")
          features = np.mean(np.array([ features, column_tensors[i] ]), axis=0)
      elif combination_method == 'weighted_l1':
          if data_purpose not in no_output:
            print("Combining with Weighted L1.")
          features = np.absolute(np.subtract(features, column_tensors[i]))
      elif combination_method == 'weighted_l2':
          if data_purpose not in no_output:
            print("Combining with Weighted L2.")
          features = np.square(np.absolute(np.subtract(features, column_tensors[i])))
      elif combination_method == 'concatenate':
          if data_purpose not in no_output:
            print("Combining with Concatenate.")
          features = np.concatenate([features, column_tensors[i]], 1)
          
  return features, original_labels, indexed_labels, keys

def train_and_eval(model_dir, model_type, train_steps, train_data, test_data, train_embeddings_file_name, test_embeddings_file_name, positive_labels, combination_method, method):
  """Train and evaluate the model."""
  
  index_map, weights = wvd.load(train_embeddings_file_name)
  #Get positive labels
  positive_labels = positive_labels.split(',')
  
  print("reading data...")
  train_file_name = train_data 
  df_train = pd.read_table(train_file_name, dtype={'node1':str, 'node2':str})
  df_train = df_train.sample(frac=1)

  # remove NaN elements
  df_train = df_train.dropna(how='any', axis=0)
  
  df_train[LABEL_COLUMN] = (
      df_train["label"].apply(lambda x: label_func(x, positive_labels))).astype(int)

  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print("model directory = %s" % model_dir)
  
  train_x, _, train_y, _ = get_input(df_train, weights, index_map, combination_method)
  
  print("\nBuilding model...")
  m = build_estimator(model_dir, model_type, weights, index_map, combination_method)
  
  print("\nTraining model...")
  if model_type == "regressor":
      m.fit(train_x, train_y, n_epoch=train_steps, show_metric=True, snapshot_epoch=False)
  
  print("\nTesting model...")
  index_map, weights = wvd.load(test_embeddings_file_name)
  
  print("reading data...")
  test_file_name = test_data
  df_test = pd.read_table(test_file_name, dtype={'node1':str, 'node2':str})
  df_test = df_test.sample(frac=1)

  # remove NaN elements
  df_test = df_test.dropna(how='any', axis=0)
  
  df_test[LABEL_COLUMN] = (
      df_test["label"].apply(lambda x: label_func(x, positive_labels))).astype(int)
  
  if model_type == "regressor":
    test_x, test_original_y, test_index_y, test_original_x = get_input(df_test, weights, index_map, combination_method, data_purpose='test')
    node_sets = get_node_sets(test_original_x, test_original_y)
    
    print("\nPredicting:")
    model_predictions = m.predict(test_x)
    model_predictions = list(model_predictions)
    #Covert back to 1 and 0
    predictions = []
    model_predictions_probs = []
    for prediction in model_predictions:
        predictions.append(prediction[1]) #non-thresholded value of positve class
        model_predictions_probs.append(prediction[1])
        
    k = int(len([i for i in test_original_y if i == 1]) * 0.3)
    do_evaluations([x for x in test_original_x], [y for y in test_original_y], [p for p in predictions], k, node_sets, 
                   positive_labels, model=m, weights=weights, index_map=index_map, combination_method=combination_method)
    #Uncomment to log ranked links
    #log_predictions([x for x in test_original_x], [y for y in test_original_y], [p for p in predictions], k, node_sets, 
    #               positive_labels, model=m, weights=weights, index_map=index_map, combination_method=combination_method,
    #               outfilename=combination_method, method=method)
    
    
def do_evaluations(test_x, test_y, predictions, k, node_sets, positive_labels=None, model=None, graph=None, vertices=None,
                   weights=None, index_map=None, combination_method=None, bipartite=False, sim_ind=None, error_anlysis=False):
    #Area under ROC
    roc_auc = metrics.roc_auc_score(test_y, predictions, average='micro')
    print("ROC AUC: %s" % roc_auc)
    
    #Area under Precision-recall curve
    avg_prec = metrics.average_precision_score(test_y, predictions, average='micro')
    print("Overall Average precision (corresponds to AUPRC): %s" % avg_prec)
        
    predictions_cpy = [pred for pred in predictions]
    test_y_cpy = [y for y in test_y]
    test_x_cpy = [x for x in test_x]
    k_cpy = k
    
    #Mean Average Precision (MAP)
    print("Calculating Averaged R-Precision and MAP")
    total_rp = 0.0
    total_ap = 0.0
    total_rp_error = 0.0
    total_ap_error = 0.0
    no_pos_nodes = 0
    if model:
        #TODO: Check for these values being present without raising the Value error about any and all for truth value of array being ambiguous
        #assert weights != None and index_map != None and positive_labels != None and \
        #    combination_method, "If model is specified, weights, index_map, combination method and positive_labels must be given."
        for node, dict_ in node_sets.iteritems():
            map_df_test = pd.DataFrame(dict_)
            map_df_test[LABEL_COLUMN] = (
                map_df_test["label"].apply(lambda x: label_func(x, positive_labels))).astype(int)
            test_x, test_original_y, test_index_y, test_original_x = get_input(map_df_test, weights, index_map, combination_method, data_purpose='map')

            model_predictions = model.predict(test_x)
            model_predictions = list(model_predictions)
            predictions = []
            for prediction in model_predictions:
                predictions.append(prediction[1]) #non-thresholded value of positve class
            test_original_y = [y for y in test_original_y]
            node_pos_cnt = len([i for i in test_original_y if i ==1])
            pos_indices = [ind for ind, i in enumerate(test_original_y) if i ==1]
            rp = r_precision([p for p in predictions], pos_indices, node_pos_cnt, lambda ind,rel_indices: ind in rel_indices)
            
            if rp < 1.0 and node_pos_cnt != 0:
                total_rp_error += (1.0 - rp)
                if error_anlysis:
                    top_k = get_top_k([p for p in predictions], node_pos_cnt)
                    print("Pos Gold indices: %s. Pos predicted indices: %s. RP: %s." % (pos_indices, [ind for ind in top_k], rp))
            
            ap = metrics.average_precision_score(np.array(test_original_y), np.array(predictions), average='micro')
            if ap < 1.0 and node_pos_cnt != 0:
                total_ap_error += (1.0 - ap)
                if error_anlysis:
                    top_k = get_top_k([p for p in predictions], len(predictions))
                    print("Pos Gold indices: %s. Pos predicted indices: %s. AP: %s." % (pos_indices, [ind for ind in top_k], ap))
            if str(ap) == 'nan':
                ap = 0.0
            
            if node_pos_cnt < 1:
                no_pos_nodes += 1 #This node had no positive labels
            total_rp += rp
            total_ap += ap
    elif graph:
        assert sim_ind, "Similarity Index must be specified."
        if bipartite:
            print("Evaluations of graph. Processing graph as bipartite.")
        for node, dict_ in node_sets.iteritems():
            node1_lst = dict_['node1']
            node2_lst = dict_['node2']
            label_lst = []
            for l in dict_['label']:
                if l == 'O':
                    label_lst.append(0)
                else:
                    label_lst.append(1)
            assert len(node1_lst) == len(node2_lst) == len(label_lst), "Nodes and labels lists of unequal length: %s, %s, %s" % (len(node1_lst), len(node2_lst), len(label_lst))
            predictions = []
            for entity1, entity2 in zip(node1_lst, node2_lst):
                entity1_set = set(graph[entity1])
                entity2_set = set(graph[entity2])
                
                if bipartite:
                    assert vertices, "Vertices must be passed with bipartite graphs."
                    #Get neighbours of neighbours of this node to use, so do new entity2_set
                    entity2_lst = []
                    for node in entity2_set:
                        if node in vertices:
                            entity = str(vertices[node])
                            entity2_lst += graph[entity]
                    entity2_set = set(entity2_lst)
                    
                #Calculate similarity index 
                assert sim_ind in ['common_neighbours', 'jaccard_coefficient', 'adamic_adar'], "Invalid similarity index %s" % sim_ind
                cn = len(entity2_set.intersection(entity1_set)) #Calculate common neighbours which all metrics use
                if sim_ind == 'common_neighbours':
                    si = cn
                elif sim_ind == 'jaccard_coefficient':
                    neighbours_union_len = len(entity2_set.union(entity1_set))
                    if neighbours_union_len > 0:
                        si = cn / float(neighbours_union_len)
                    else:
                        si = 0.0
                elif sim_ind == 'adamic_adar':
                    if cn > 1:
                        si = 1.0/math.log(cn)
                    else:
                        si = 0.0
                predictions.append(float(si))
            
            node_pos_cnt = len([i for i in label_lst if i ==1])
            pos_indices = [ind for ind, i in enumerate(label_lst) if i ==1]
            rp = r_precision([p for p in predictions], pos_indices, node_pos_cnt, lambda ind,rel_indices: ind in rel_indices)
            
            if rp < 1.0 and node_pos_cnt != 0:
                total_rp_error += (1.0 - rp)
                if error_anlysis:
                    top_k = get_top_k([p for p in predictions], node_pos_cnt)
                    print("\nPos Gold indices: %s. Pos predicted indices: %s. RP: %s" % (pos_indices, [ind for ind in top_k], rp))
                    
            ap = metrics.average_precision_score(np.array(label_lst), np.array(predictions), average='micro')
            if ap < 1.0 and node_pos_cnt != 0:
                total_ap_error += (1.0 - ap)
                if error_anlysis:
                    print("labels: %s. Predictions: %s. AP: %s." % (label_lst, predictions, ap))
            if str(ap) == 'nan':
                ap = 0.0
            
            if node_pos_cnt < 1:
                rp = 0.0
                no_pos_nodes += 1 #This node had no positive labels
            total_rp += rp
            total_ap += ap
            
    print("----- Total RP error: %s" % total_rp_error)
    print("----- Total rp: %s. Viable Nodes: %s"  % (total_rp, len(node_sets)-no_pos_nodes))
    print("Mean R-Precision: %s (%s/%s nodes with no positives.)" % (total_rp/(len(node_sets)-no_pos_nodes), no_pos_nodes, len(node_sets)) )
    
    print("----- Total AP error: %s" % total_ap_error)
    print("----- Total ap: %s. Viable Nodes: %s"  % (total_ap, len(node_sets)-no_pos_nodes))
    print("MAP: %s (%s/%s nodes with no positives.)" % (total_ap/(len(node_sets)-no_pos_nodes), no_pos_nodes, len(node_sets)) )
    
    #Precision @ k
    print("Calculating Precision @ k (k = %s)" % k_cpy)
    p_at_k = calculate_top_k(predictions_cpy, test_y_cpy, k_cpy, test_x_cpy)
    print("Precision @ %s: %s" % (k_cpy, p_at_k))
    
def log_predictions(test_x, test_y, predictions, k, node_sets, positive_labels=None, model=None, graph=None, vertices=None,
                   weights=None, index_map=None, combination_method=None, bipartite=False, sim_ind=None, error_anlysis=False,
                   outfilename='ranked_edges', method=None):
    #Log the predictions
    
    #All edges
    predictions_cpy = [pred for pred in predictions]
    test_y_cpy = [y for y in test_y]
    test_x_cpy = [x for x in test_x]
    ordered_preds = get_top_k(predictions, len(predictions))
    output = "link\tscore\tgold label\n"
    for index in ordered_preds:
        output += "{}\t{}\t{}\n".format(test_x[index], predictions[index], test_y[index])
    filename = "ranked-edges-all-{}-{}.tsv".format(method, outfilename) if method else "ranked-edges-all-{}.tsv".format(outfilename)
    all_edges_file = open(filename, 'w')
    all_edges_file.write(output)
    all_edges_file.close()
    
    #Node edges
    if model:
        #TODO: Check for these values being present without raising the Value error about any and all for truth value of array being ambiguous
        #assert weights != None and index_map != None and positive_labels != None and \
        #    combination_method, "If model is specified, weights, index_map, combination method and positive_labels must be given."
        rp_error_lst = []
        ap_error_lst = []
        node_lst = []
        output = "node name/link\tscore\tgold label\n"
        for node, dict_ in node_sets.iteritems():
            node_lst.append(node)
            output += "\n{}-\n".format(node)
            
            map_df_test = pd.DataFrame(dict_)
            map_df_test[LABEL_COLUMN] = (
                map_df_test["label"].apply(lambda x: label_func(x, positive_labels))).astype(int)
            test_x, test_original_y, test_index_y, test_original_x = get_input(map_df_test, weights, index_map, combination_method, data_purpose='map')

            model_predictions = model.predict(test_x)
            model_predictions = list(model_predictions)
            predictions = []
            for prediction in model_predictions:
                predictions.append(prediction[1]) #non-thresholded value of positve class
            ordered_preds = get_top_k(predictions, len(predictions))
            
            for index in ordered_preds:
                output += "{}\t{}\t{}\n".format(test_original_x[index], predictions[index], test_original_y[index])
                
            #record, rank and log the AP and RP errors
            node_pos_cnt = len([i for i in test_original_y if i ==1])
            pos_indices = [ind for ind, i in enumerate(test_original_y) if i ==1]
            rp = r_precision([p for p in predictions], pos_indices, node_pos_cnt, lambda ind,rel_indices: ind in rel_indices)
            ap = metrics.average_precision_score(np.array(test_original_y), np.array(predictions), average='micro')
            if str(ap) == 'nan':
                ap = 2.0
            if node_pos_cnt < 1:
                rp = 2.0
            rp_error_lst.append((1.0 - rp))
            ap_error_lst.append((1.0 - ap))
        
        filename = "ranked-edges-nodes-{}-{}.tsv".format(method, outfilename) if method else "ranked-edges-nodes-{}.tsv".format(outfilename)
        node_edges_file = open(filename, 'w')
        node_edges_file.write(output)
        node_edges_file.close()
        
        #Order these by Averaged R-precision errors or MAP errors (but not both!)
        #ordered_errors = get_top_k(rp_error_lst, len(rp_error_lst))
        ordered_errors = get_top_k(ap_error_lst, len(ap_error_lst))
        output = "node\tR-Precision error\tAve-Precision Error\n"
        for index in ordered_errors:
            output += "{}\t{}\t{}\n".format(node_lst[index], rp_error_lst[index], ap_error_lst[index])
        filename = "ranked-rp_errors-{}-{}.tsv".format(method, outfilename) if method else "ranked-rp_errors-{}.tsv".format(outfilename)
        ranked_rp_file = open(filename, 'w')
        ranked_rp_file.write(output)
        ranked_rp_file.close()  
    elif graph:
        assert sim_ind, "Similarity Index must be specified."
        if bipartite:
            print("Evaluations of graph. Processing graph as bipartite.")
            
        rp_error_lst = []
        ap_error_lst = []
        node_lst = []
        output = "node name/link\tscore\tgold label\n"
        for node, dict_ in node_sets.iteritems():
            node_lst.append(node)
            output += "\n{}-\n".format(node)
            node1_lst = dict_['node1']
            node2_lst = dict_['node2']
            label_lst = []
            for l in dict_['label']:
                if l == 'O':
                    label_lst.append(0)
                else:
                    label_lst.append(1)
            assert len(node1_lst) == len(node2_lst) == len(label_lst), "Nodes and labels lists of unequal length: %s, %s, %s" % (len(node1_lst), len(node2_lst), len(label_lst))
            predictions = []
            test_x = []
            for entity1, entity2 in zip(node1_lst, node2_lst):
                entity1_set = set(graph[entity1])
                entity2_set = set(graph[entity2])
                test_x.append("{}::{}".format(entity1, entity2))
                
                if bipartite:
                    assert vertices, "Vertices must be passed with bipartite graphs."
                    #Get neighbours of neighbours of this node to use, so do new entity2_set
                    entity2_lst = []
                    for node in entity2_set:
                        if node in vertices:
                            entity = str(vertices[node])
                            entity2_lst += graph[entity]
                    entity2_set = set(entity2_lst)
                    
                #Calculate similarity index 
                assert sim_ind in ['common_neighbours', 'jaccard_coefficient', 'adamic_adar'], "Invalid similarity index %s" % sim_ind
                cn = len(entity2_set.intersection(entity1_set)) #Calculate common neighbours which all metrics use
                if sim_ind == 'common_neighbours':
                    si = cn
                elif sim_ind == 'jaccard_coefficient':
                    neighbours_union_len = len(entity2_set.union(entity1_set))
                    if neighbours_union_len > 0:
                        si = cn / float(neighbours_union_len)
                    else:
                        si = 0.0
                elif sim_ind == 'adamic_adar':
                    if cn > 1:
                        si = 1.0/math.log(cn)
                    else:
                        si = 0.0
                predictions.append(float(si))
            
            ordered_preds = get_top_k(predictions, len(predictions))
            for index in ordered_preds:
                output += "{}\t{}\t{}\n".format(test_x[index], predictions[index], label_lst[index])
            
            #record, rank and log the AP and RP errors
            node_pos_cnt = len([i for i in label_lst if i ==1])
            pos_indices = [ind for ind, i in enumerate(label_lst) if i ==1]
            rp = r_precision([p for p in predictions], pos_indices, node_pos_cnt, lambda ind,rel_indices: ind in rel_indices)
            ap = metrics.average_precision_score(np.array(label_lst), np.array(predictions), average='micro')
            if str(ap) == 'nan':
                ap = 2.0
            if node_pos_cnt < 1:
                rp = 2.0
            rp_error_lst.append((1.0 - rp))
            ap_error_lst.append((1.0 - ap))
        
        filename = "ranked-edges-nodes-{}-{}.tsv".format(method, outfilename) if method else "ranked-edges-nodes-{}.tsv".format(outfilename)
        node_edges_file = open(filename, 'w')
        node_edges_file.write(output)
        node_edges_file.close()
            
        #Order these by Averaged R-precision errors or MAP errors (but not both!)
        #ordered_errors = get_top_k(rp_error_lst, len(rp_error_lst))
        ordered_errors = get_top_k(ap_error_lst, len(ap_error_lst))
        output = "node\tR-Precision error\tAve-Precision Error\n"
        for index in ordered_errors:
            output += "{}\t{}\t{}\n".format(node_lst[index], rp_error_lst[index], ap_error_lst[index])
        filename = "ranked-rp_errors-{}-{}.tsv".format(method, outfilename) if method else "ranked-rp_errors-{}.tsv".format(outfilename)
        ranked_rp_file = open(filename, 'w')
        ranked_rp_file.write(output)
        ranked_rp_file.close()  
        
def r_precision(predictions, gold, rel_no, rel_fn):
    if rel_no == 0:
        return 0.0
    rel_total = 0
    top_k = get_top_k(predictions, rel_no)
    for ind in top_k:
        if rel_fn(ind, gold):
            rel_total += 1
    return rel_total/float(rel_no)
    
def use_similarity_indices(test_data, positive_labels, bipartite=False):
    if str(bipartite).lower() == 'false':
        bipartite = False
    #Get positive labels
    positive_labels = positive_labels.split(',')
    
    print("reading data...")
    test_file_name = test_data
    df_test = pd.read_table(test_file_name, dtype={'node1':str, 'node2':str})
    df_test = df_test.sample(frac=1)

    # remove NaN elements
    df_test = df_test.dropna(how='any', axis=0)
    
    df_test[LABEL_COLUMN] = (
        df_test["label"].apply(lambda x: label_func(x, positive_labels))).astype(int)
    
    label_values = df_test[LABEL_COLUMN].values
    test_original_y = np.array(label_values)
    
    col_keys = []
    for i in COLUMNS:
        words = [value for value in df_test[i].values]
        col_keys.append(words)
        
    test_original_x = []
    for entity1, entity2 in zip(col_keys[0], col_keys[1]):
        test_original_x.append("%s::%s" % (entity1, entity2))
        
    if not bipartite:
        print("Assuming graph not bipartite.")
        similarity_indices(test_original_x, test_original_y)
    else:
        print("Processing graph as bipartite.")
        similarity_indices_bipartite(test_original_x, test_original_y, matador_separation)
    

def label_func(x, positive_labels):
    for ind, label in enumerate(positive_labels, 1):
        if label in x:
            return ind
    return 0

def get_node_sets(test_x, test_y):
    node_sets = {}
    for node_pair, label in zip(test_x, test_y):
        entity1 = node_pair.split('::')[0]
        entity2 = node_pair.split('::')[1]
        if entity1 not in node_sets:
            if label == 1:
                node_sets[entity1] = ([node_pair], [])
            else:
                node_sets[entity1] = ([], [node_pair])
        else:
            if label == 1:
                node_sets[entity1][0].append(node_pair)
            else:
                node_sets[entity1][1].append(node_pair)
                
        if entity2 not in node_sets:
            if label == 1:
                node_sets[entity2] = ([node_pair], [])
            else:
                node_sets[entity2] = ([], [node_pair])
        else:
            if label == 1:
                node_sets[entity2][0].append(node_pair)
            else:
                node_sets[entity2][1].append(node_pair)
    
    for entity, tup in node_sets.iteritems():
        node_sets[entity] = {'node1' : pd.Series([key.split('::')[0] for key in tup[0]] + [key.split('::')[0] for key in tup[1]]),
                             'node2' : pd.Series([key.split('::')[1] for key in tup[0]] + [key.split('::')[1] for key in tup[1]]), 
                             'label' : pd.Series((['I-LINK'] * len(tup[0])) + (['O'] * len(tup[1])) )}
    return node_sets

def get_top_k(predictions, k):
    top_k = [] #Indices of top k results
    
    if len(Counter(predictions).most_common(2)) < 2:
        for i in range(k):
            top_k.append(random.randrange(len(predictions)))
        return top_k
    else:
        if not any(predictions):
            print("Preditions: %s. counter mc is: %s." % (predictions, Counter(predictions).most_common(2)))
        
    report = False
    if k > 50000:
        #Set up reporting mechanism
        report = True
        report_quarter = False
        report_half = False
        report_three_quarter = False
        quarter = int(.25 * k)
        half = int(.5 * k)
        three_quarter = int(.75 * k)

    sorted_preds = sorted(predictions, reverse=True)
    start = 0
    prev_value = -1
    #top_5_scores = []
    for i in range(k):
        if report:
            if not report_quarter and i >= quarter:
                print("Quarter complete.")
                report_quarter = True
            if not report_half and i >= half:
                print("Half complete.")
                report_half = True
            if i >= three_quarter and not report_three_quarter:
                print("Three quarters complete.")
                report_three_quarter = True
                report = False
                
        if sorted_preds[i] != prev_value:
            start = 0
        max_ind = predictions.index(sorted_preds[i], start)
        prev_value = sorted_preds[i] 
        start = max_ind + 1
        top_k.append(max_ind)
        if len(top_k) == len(predictions):
            break
    if len(predictions) > k and len(top_k) != k:
        print("ERROR: Length of top-k is %s but there are %s predictions." % (len(top_k), len(predictions)))
    return top_k

def calculate_top_k(predictions, gold, k, test_egs):
    preds_cpy = [p for p in predictions]
    print("Start top k: {}".format(datetime.now().strftime('%H:%M:%S')))
    top_k = get_top_k([p for p in predictions], k)
    print("End top k: {}".format(datetime.now().strftime('%H:%M:%S')))
        
    correct_cnt = 0
    correct_egs = []
    wrong_egs = []
    for index in top_k:
        if gold[index] == 1:
            correct_cnt += 1
            correct_egs.append((test_egs[index], preds_cpy[index]))
        else:
            wrong_egs.append((test_egs[index], preds_cpy[index]))
    correct_preds_output = ""
    for eg in correct_egs:
        correct_preds_output += "{}\n".format(eg)
    incorrect_preds_output = ""
    for eg in wrong_egs:
        incorrect_preds_output += "{}\n".format(eg)
        
    #Uncomment for some error logging
    #cor_file = open("correct_pred.txt", 'w')
    #cor_file.write(correct_preds_output)
    #cor_file.close()
    #incor_file = open("incorrect_pred.txt", 'w')
    #incor_file.write(incorrect_preds_output)
    #incor_file.close()
        
    return correct_cnt/float(k)

def similarity_indices_bipartite(test_egs, test_labels, separation_fn):
    #Read in vertices and node indexes
    vertices = {}
    vertices_file = open('vertices.txt', 'r')
    for line in vertices_file:
        line = line.split()
        vertices[line[0]] = line[1]
            
    vertex_set1, vertex_set2 = separation_fn(vertices)

    #Read in graph adjacency list
    graph = {}
    train_edges = {}
    graph_file = open('graph.adjlist', 'r')
    for line in graph_file:
        line = line.split()
        if line[0] in vertices:
            node = vertices[line[0]]
            rel_nodes = [vertices[node_ind] for node_ind in line[1:]] 
            graph[node] = [node_ for node_ in line[1:]]
            for rel_node in rel_nodes:
                train_edges["%s::%s" % (node, rel_node)] = 1 #Dict not really needed here but indexing in a dict is much faster than a list.
        else:
            print("Node index: %s not in vertices." % line[0])
            
    jc_predictions = []
    cn_predictions = []
    aa_predictions = []
    for node_pair, label in zip(test_egs, test_labels):
        entity1 = node_pair.split('::')[0]
        entity2 = node_pair.split('::')[1]
        
        entity1_set = set(graph[entity1])
        entity2_set = set(graph[entity2])

        #Get neighbours of neighbours of this node to use
        entity2_lst = []
        for node in entity2_set:
            if node in vertices:
                entity = str(vertices[node])
                entity2_lst += graph[entity]
                
        entity2_set = set(entity2_lst)
        #Common Neighbours and Jaccard Co-efficient
        if (len(entity2_set) + len(entity1_set)) > 0:
            jc = len(entity2_set.intersection(entity1_set)) / float(len(entity2_set) + len(entity1_set))
            cn = len(entity2_set.intersection(entity1_set))
        else:
            jc = 0.0
            cn = 0.0
            
        #Adamic-Adar, using common neighbours as the shared feature
        if len(entity2_set.intersection(entity1_set)) > 1: #log(1) not defined
            aa = 1/math.log(len(entity2_set.intersection(entity1_set)))
        else:
            aa = 0.0

        jc_predictions.append(jc)
        cn_predictions.append(cn)
        aa_predictions.append(aa)

    node_sets = get_node_sets(test_egs, test_labels)
    k = int(len([i for i in test_labels if i == 1]) * 0.9)
    for name, predictions in zip(['common_neighbours', 'jaccard_coefficient', 'adamic_adar'], [cn_predictions, jc_predictions, aa_predictions]):
        print("\nEvaluating %s" % name)
        do_evaluations(test_egs, test_labels, predictions, k, node_sets, graph=graph, vertices=vertices, bipartite=True, sim_ind=name)
        #Uncomment to log ranked links
        #log_predictions(test_egs, test_labels, predictions, k, node_sets, graph=graph, vertices=vertices, bipartite=True, sim_ind=name, outfilename=name)
    
#In MATADOR dataset, all Chemical identifiers can be integers but no proteins can be
def matador_separation(vertices):
    proteins = {}
    chemicals = {}
    for vertex_ind, vertex_name in vertices.iteritems():
        try:
            int(vertex_name)
            chemicals[vertex_name] = 1
        except:
            proteins[vertex_name] = 1
            
    print("For MATADOR dataset: %s proteins and %s chemicals." % (len(proteins), len(chemicals)))
    return proteins, chemicals
    
#Similarity indices for non-bipartite graphs
def similarity_indices(test_egs, test_labels):
    #Read in vertices and node indexes
    vertices = {}
    vertices_file = open('vertices.txt', 'r')
    for line in vertices_file:
        line = line.split()
        vertices[line[0]] = line[1]

    #Read in graph adjacency list
    graph = {}
    train_edges = {}
    graph_file = open('graph.adjlist', 'r')
    for line in graph_file:
        line = line.split()
        if line[0] in vertices:
            node = vertices[line[0]]
            rel_nodes = [vertices[node_ind] for node_ind in line[1:]] 
            graph[node] = [node_ for node_ in line[1:]]
            for rel_node in rel_nodes:
                train_edges["%s::%s" % (node, rel_node)] = 1 #Dict not really needed here but indexing in a dict is much faster than a list.
        else:
            print("Node index: %s not in vertices." % line[0])
            
    jc_predictions = []
    cn_predictions = []
    aa_predictions = []
    for node_pair, label in zip(test_egs, test_labels):
        entity1 = node_pair.split('::')[0]
        entity2 = node_pair.split('::')[1]
        
        entity1_set = set(graph[entity1])
        entity2_set = set(graph[entity2])
        
        cn = len(entity2_set.intersection(entity1_set)) # calculate Common neighbours for use by all
        neighbours_union_len = len(entity2_set.union(entity1_set))
        #Common Neighbours and Jaccard Co-efficient
        if neighbours_union_len > 0:
            jc = cn / float(neighbours_union_len)
        else:
            jc = 0.0
            
        #Adamic-Adar, using common neighbours as the shared feature
        if cn > 1: #log(1) not defined
            aa = 1/math.log(cn)
        else:
            aa = 0.0

        jc_predictions.append(jc)
        cn_predictions.append(float(cn))
        aa_predictions.append(aa)

    node_sets = get_node_sets(test_egs, test_labels)
    k = int(len([i for i in test_labels if i == 1]) * 0.9)
    for name, predictions in zip(['common_neighbours', 'jaccard_coefficient', 'adamic_adar'], [cn_predictions, jc_predictions, aa_predictions]): #  
        print("\nEvaluating %s" % name)
        do_evaluations(test_egs, test_labels, predictions, k, node_sets, graph=graph, vertices=vertices, sim_ind=name)
        #Uncomment to log ranked links
        #log_predictions(test_egs, test_labels, predictions, k, node_sets, graph=graph, vertices=vertices, sim_ind=name, outfilename=name)
    
FLAGS = None
def main(_):
    if FLAGS.model_type == "similarity_indices":
        use_similarity_indices(FLAGS.test_data, FLAGS.positive_labels, FLAGS.graph_bipartite)
    else:
        train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps, FLAGS.train_data, FLAGS.test_data, FLAGS.train_embeddings_data, FLAGS.test_embeddings_data, FLAGS.positive_labels, FLAGS.combination_method, 
                       FLAGS.method)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=500,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  parser.add_argument(
      "--train_embeddings_data",
      type=str,
      default="",
      help="Path to the pre-trained embeddings file for training."
  )
  parser.add_argument(
      "--test_embeddings_data",
      type=str,
      default="",
      help="Path to the pre-trained embeddings file for testing."
  )
  parser.add_argument(
      "--positive_labels",
      type=str,
      default="I-LINK",
      help="Label of positive classes in data, separated by comma."
  )
  parser.add_argument(
      "--combination_method",
      type=str,
      default="concatenate",
      help="How the features should be combined by the model."
  )
  parser.add_argument(
      "--graph_bipartite",
      type=str,
      default=False,
      help="Process graph as bipartitie or not for Common Neighbours."
  )
  parser.add_argument(
      "--method",
      type=str,
      default="",
      help="Method used to create embeddings."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

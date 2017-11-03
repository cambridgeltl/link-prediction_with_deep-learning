#!/bin/sh

for i in 1 #2 3
do
    random_slice=True
    time_slice=False
    
    if [ "$random_slice" = True ]
    then
        echo "\n----------------------------------------------------------"
        echo "----------------------------------------------------------"
        echo "----------------------------------------------------------"
        echo "----------------------------------------------------------"
        echo "Experiments run count ${i}"
            
        for dataset in matador #lion_pubmed biogrid_genetic
        do
            echo "\n----------------------------------------------------------"
            echo "----------------------------------------------------------"
            echo "----------------------------------------------------------"
            echo "Using dataset ${dataset}."
            if [ "$dataset" = "matador" ]
            then
                datapath='data/MATADOR/matador.tsv'
                embeddingsshortname='dti'
                bipartite='True'
                ci='0:3'
            elif  [ "$dataset" = "biogrid_genetic" ]
            then
                datapath='data/BioGRID/BioGRID-All-with_dates.tsv'
                embeddingsshortname='biogrid-genetic'
                bipartite='False'
                ci='0:1'
            elif  [ "$dataset" = "lion_pubmed" ]
            then
                datapath='data/LION-PubMed/edges.tsv'
                embeddingsshortname='lion-pubmed'
                bipartite='False'
                ci='0:1'
            fi
            
            #Setup experiment
            python 'experiment_setup.py' -f ${datapath} -s '20:0:80' -il '50:50' -a '' --maintain-connection True -ci ${ci} --graph_bipartite ${bipartite}
            
            #create representations
            for method in similarity_indices #deepwalk node2vec line
            do
                echo "\n----------------------------------------------------------"
                echo "----------------------------------------------------------"
                echo "Method: ${method}."
                if [ $method = 'similarity_indices' ]
                then
                    echo "Doing Link prediction with Similarity Indices."
                    echo "Bipartite: " ${bipartite}
                    python 'Models/feed_forward/tensorflow/link_prediction.py' --train_data "train.tsv" --test_data "test.tsv" --train_steps 0 --model_type "similarity_indices" --graph_bipartite ${bipartite}
                else
                    echo "Doing Link prediction with neural network model."
                    
                    if [ $method = 'deepwalk' ]
                    then
                        #Create representations
                        echo "Creating node representations with DeepWalk"
                        deepwalk --input 'test_adj_mat.adjlist' --output "test_${embeddingsshortname}.embeddings" --representation-size 100 --walk-length 40 --window-size 10 --number-walks 40
                    elif [ $method = 'node2vec' ]
                    then
                        #Create representations
                        echo "Creating node representations with node2vec"
                        #Inefficient node2vec
                        #python 'node2vec/src/main.py' --input 'test_adj_mat.edgelist' --output "test_${embeddingsshortname}.embeddings" --dimensions 100 --p 2 --q 4 --walk-length 40 --window-size 10
                        #Efficient node2vec
                        ./node2vec_cpp/node2vec -i:'test_adj_mat.edgelist' -o:"test_${embeddingsshortname}.embeddings" -l:40 -d:100 -p:2 -q:4 -k:10 -v
                    elif [ $method = 'line' ]
                    then
                        echo "Setting up experiment for LINE"
                        #Create representations
                        echo "Creating node representations with LINE"
                        LINE/linux/line -train 'test_adj_mat.line' -output "test_${embeddingsshortname}-order1.embeddings" -size 50 -order 1 -samples 10000 #HALVE SO THAT COMBINED CAN HAVE DESIRED DIM
                        LINE/linux/line -train 'test_adj_mat.line' -output "test_${embeddingsshortname}-order2.embeddings" -size 50 -order 2 -samples 10000 #HALVE SO THAT COMBINED CAN HAVE DESIRED DIM
                        #Concatenate and normalise as recomended in paper
                        python line_combine.py -i1 "test_${embeddingsshortname}-order1.embeddings" -i2 "test_${embeddingsshortname}-order2.embeddings" -o "test_${embeddingsshortname}.embeddings"
                    else
                        echo "UNKNOWN NODE CREATION METHOD!"
                    fi
                    
                    #Create the modified embeddings to change from node indices to node names
                    echo "Creating modified embeddings."
                    python 'create_modified_embeddings.py' -f "test_${embeddingsshortname}.embeddings" -o  "test_modified_${embeddingsshortname}.embeddings"

                    #Convert the embeddings to .bin format
                    echo "Converting embeddings to .bin format."
                    python 'wvlib/convert.py' -i sdv "test_modified_${embeddingsshortname}.embeddings" "test_modified_${embeddingsshortname}.embeddings.bin"
                    
                    
                    for combination_method in concatenate average #hadamard weighted_l2 weighted_l1
                    do
                        echo "\n----------------------------------------------------------"
                        echo "Training regressor with ${combination_method}." 
                        #Train model
                        python 'Models/feed_forward/tensorflow/link_prediction.py' --train_data "train.tsv" --test_data "test.tsv" --train_embeddings_data "test_modified_${embeddingsshortname}.embeddings.bin" --test_embeddings_data "test_modified_${embeddingsshortname}.embeddings.bin" --train_steps 7 --model_type "regressor" --combination_method ${combination_method} --method ${method}  
                    done
                fi
            done
        done
    fi
    ################################################################################################################################################
    if [ "$time_slice" = True ]
    then
        echo "\nLiterature sliced experiments."
        #Literature sliced experiments
        echo "\n----------------------------------------------------------"
        echo "----------------------------------------------------------"
        echo "----------------------------------------------------------"
        echo "----------------------------------------------------------"
        echo "Experiments run count ${i}"
            
        for dataset in biogrid_genetic #lion_pubmed
        do
            echo "\n----------------------------------------------------------"
            echo "----------------------------------------------------------"
            echo "----------------------------------------------------------"
            echo "Using dataset ${dataset}."
            if  [ "$dataset" = "biogrid_genetic" ]
            then
                datapath='data/BioGRID/BioGRID-All-with_dates.tsv'
                embeddingsshortname='biogrid-genetic'
                bipartite='False'
                ci='0:1:2'
                p1_end_date='2014'
                p2_start_date='2013'
                p2_end_date='2014'
                p3_start_date='2015'
            elif  [ "$dataset" = "lion_pubmed" ]
            then
                datapath='data/LION-PubMed/edges.tsv'
                embeddingsshortname='lion-pubmed'
                bipartite='False'
                ci='0:1:2'
                p1_end_date='2002'
                p2_start_date='2001'
                p2_end_date='2002'
                p3_start_date='2003'
            fi
            
            #Setup experiment
            python 'experiment_setup.py' -f ${datapath} -cl 'entity1,entity2,date' -a 'date' -sc "p1:date,<=,${p1_end_date}::p2:date,>=,${p2_start_date}|date,<=,${p2_end_date}::p3:date,>=,${p3_start_date}" -is 'p1' -tr 'p2' -te 'p3' --maintain-connection True -ci ${ci}
            
            #create representations
            for method in similarity_indices node2vec #deepwalk line
            do
                echo "\n----------------------------------------------------------"
                echo "----------------------------------------------------------"
                echo "Method: ${method}."
                if [ $method = 'similarity_indices' ]
                then
                    echo "Doing Link prediction with Similarity Indices."
                    echo "Bipartite: " ${bipartite}
                    python 'Models/feed_forward/tensorflow/link_prediction.py' --train_data "train.tsv" --test_data "test.tsv" --train_steps 0 --model_type "similarity_indices" --graph_bipartite ${bipartite}
                else
                    echo "Doing Link prediction with neural network model."
                    
                    if [ $method = 'deepwalk' ]
                    then
                        #Create representations
                        echo "Creating node representations with DeepWalk"
                        deepwalk --input 'test_adj_mat.adjlist' --output "test_${embeddingsshortname}.embeddings" --representation-size 100 --walk-length 40 --window-size 10 --number-walks 40
                    elif [ $method = 'node2vec' ]
                    then
                        #Create representations
                        echo "Creating node representations with node2vec"
                        #Inefficient node2vec
                        #python 'node2vec/src/main.py' --input 'test_adj_mat.edgelist' --output "test_${embeddingsshortname}.embeddings" --dimensions 100 --p 2 --q 4 --walk-length 40 --window-size 10 
                        #Efficient? node2vec
                        ./node2vec_cpp/node2vec -i:'test_adj_mat.edgelist' -o:"test_${embeddingsshortname}.embeddings" -l:40 -d:100 -p:2 -q:4 -k:10 -v
                    elif [ $method = 'line' ]
                    then
                        #Create representations
                        echo "Creating node representations with LINE"
                        LINE/linux/line -train 'test_adj_mat.line' -output "test_${embeddingsshortname}-order1.embeddings" -size 50 -order 1 -samples 10000 #HALVE SO THAT COMBINED CAN HAVE DESIRED DIM
                        LINE/linux/line -train 'test_adj_mat.line' -output "test_${embeddingsshortname}-order2.embeddings" -size 50 -order 2 -samples 10000 #HALVE SO THAT COMBINED CAN HAVE DESIRED DIM
                        #Concatenate and normalise as recomended in paper
                        python line_combine.py -i1 "test_${embeddingsshortname}-order1.embeddings" -i2 "test_${embeddingsshortname}-order2.embeddings" -o "test_${embeddingsshortname}.embeddings"
                    fi
                    
                    #Create the modified embeddings to change from node indices to node names
                    echo "Creating modified embeddings."
                    python 'create_modified_embeddings.py' -f "test_${embeddingsshortname}.embeddings" -o  "test_modified_${embeddingsshortname}.embeddings"

                    #Convert the embeddings to .bin format
                    echo "Converting embeddings to .bin format."
                    python 'wvlib/convert.py' -i sdv "test_modified_${embeddingsshortname}.embeddings" "test_modified_${embeddingsshortname}.embeddings.bin"
                    
                    
                    for combination_method in concatenate average #hadamard weighted_l2 weighted_l1
                    do
                        echo "\n----------------------------------------------------------"
                        echo "Training regressor with ${combination_method}." 
                        #Train model
                        python 'Models/feed_forward/tensorflow/link_prediction.py' --train_data "train.tsv" --test_data "test.tsv" --train_embeddings_data "test_modified_${embeddingsshortname}.embeddings.bin" --test_embeddings_data "test_modified_${embeddingsshortname}.embeddings.bin" --train_steps 7 --model_type "regressor" --combination_method ${combination_method} --method ${method}
                    done
                fi
            done
        done
    fi
done


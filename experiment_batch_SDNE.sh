#!/bin/sh

#SDNE requires a slightly different setup flow than the others.

for i in 1 #2 3
do
    echo "\n----------------------------------------------------------"
    echo "----------------------------------------------------------"
    echo "----------------------------------------------------------"
    echo "----------------------------------------------------------"
    echo "Experiments run count ${i}"
        
    for dataset in matador #biogrid_genetic lion_pubmed 
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
            bipartite=False
            ci='0:1'
        fi
        
        echo "Setting up experiment for SDNE"
        #Setup experiment
        python 'experiment_setup.py' -f ${datapath} -s '20:0:80' -il '50:50' -a '' --maintain-connection True -ci ${ci} -a '' -g 'sdne' -tgf 'train_adj_mat.sdne' -tegf 'test_adj_mat.sdne' --graph_bipartite ${bipartite}
        
        #Create representations
        echo "Creating node representations with SDNE"
        #Using Wang et al's implmentation
        #Create the config file for sdne to use
        #python create_sdne_config.py -s "${embeddingsshortname}" 
        
        #Using Goyal at al's GEM
        #Run SDNE
        python run_gem_sdne.py -i 'test_adj_mat.sdne' -o "test_modified_${embeddingsshortname}.embeddings" -vf 'vertices.txt'

        #Convert the embeddings to .bin format
        echo "Converting embeddings to .bin format."
        python 'wvlib/convert.py' -i sdv "test_modified_${embeddingsshortname}.embeddings" "test_modified_${embeddingsshortname}.embeddings.bin"
        
        
        for combination_method in concatenate average #hadamard weighted_l2 weighted_l1
        do
            echo "\n----------------------------------------------------------"
            echo "Training regressor with ${combination_method}." 
            #Train model
            python 'Models/feed_forward/tensorflow/link_prediction.py' --train_data "train.tsv" --test_data "test.tsv" --train_embeddings_data "test_modified_${embeddingsshortname}.embeddings.bin" --test_embeddings_data "test_modified_${embeddingsshortname}.embeddings.bin" --train_steps 7 --model_type "regressor" --combination_method ${combination_method} --method 'sdne'
        done
    done
done

#######################################################################################################################################################
echo "\nLiterature sliced experiments."
#Literature sliced experiments
for i in 1 #2 3
do
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
            p1_end_date='2012'
            p2_start_date='2013'
            p2_end_date='2013'
            p3_start_date='2014'
        elif  [ "$dataset" = "lion_pubmed" ]
        then
            datapath='data/LION-PubMed/edges.tsv'
            embeddingsshortname='lion-pubmed'
            bipartite='False'
            ci='0:1:2'
            p1_end_date='2000'
            p2_start_date='2001'
            p2_end_date='2002'
            p3_start_date='2003'
        fi
        
        echo "Setting up experiment for SDNE"
        #Setup experiment
        python 'experiment_setup.py' -f ${datapath} -s '20:0:80' -il '50:50' -a '' --maintain-connection True -ci ${ci} -a '' -g 'sdne' -tgf 'train_adj_mat.sdne' -tegf 'test_adj_mat.sdne' --graph_bipartite ${bipartite}
        
        #Create representations
        echo "Creating node representations with SDNE"
        #Using Wang et al's implmentation (files available on request)
        #Create the config file for sdne to use
        #python create_sdne_config.py -s "${embeddingsshortname}" 
        
        #Using Goyal at al's GEM
        #Run SDNE
        python run_gem_sdne.py -i 'test_adj_mat.sdne' -o "test_modified_${embeddingsshortname}.embeddings" -vf 'vertices.txt'

        #Convert the embeddings to .bin format
        echo "Converting embeddings to .bin format."
        python 'wvlib/convert.py' -i sdv "test_modified_${embeddingsshortname}.embeddings" "test_modified_${embeddingsshortname}.embeddings.bin"
        
        
        for combination_method in concatenate average #hadamard weighted_l2 weighted_l1
        do
            echo "\n----------------------------------------------------------"
            echo "Training regressor with ${combination_method}." 
            #Train model
            python 'Models/feed_forward/tensorflow/link_prediction.py' --train_data "train.tsv" --test_data "test.tsv" --train_embeddings_data "test_modified_${embeddingsshortname}.embeddings.bin" --test_embeddings_data "test_modified_${embeddingsshortname}.embeddings.bin" --train_steps 7 --model_type "regressor" --combination_method ${combination_method} --method 'sdne'
        done
    done
done
            
# Attacking Graph Neural Networks with Bit Flips: Weisfeiler and Leman Go Indifferent

## Installation
1) The Injectivity Bitflip Attack (IBFA) bitflip attack is seamlessly integrated in the PyQ quantization package. Hence for installation, navigate to ibfa/pyq and follow the steps documented in README.md to install PyQ.

2) After installing PyQ, new quantized models can be trained by running any of the scripts found in ibfa/pyq/graph. Make sure to adapt the paths in the script to your local environment. 
For a quick start, pre-trained quantized models can be downloaded [here](https://ucloud.univie.ac.at/index.php/s/aY5e3b6Jdyy5HTa).

3) Put the quantized models produced in the previous step either in ibfa/models or navigate to ibfa/pyq/run_bfa_{model}_{dataset}.py and adapt the paths to your local environment. They should in point to the locations where the quantized models produced in step 2. are stored.

## Execution 
From the root directory of the repository (ibfa), run
``python pyq/run_bfa_gin_molecular.py --type IBFAv1 --data ogbg-molhiv --n 1 --k 5 --sz 128``
to execute IBFA1 on GIN (or another model for which you put the path in the python script) on dataset ogbg-molhiv with k=5 BFA attack runs, batch size sz=128 and return results as average of n=1 repetitions of the experiment. 

Example output:

    IBFAv1 ogbg-molhiv 1 10 32
    Start time 03/06/2024, 13:20:04
    params 1885241
    (865, 865) (813, 813)
    IBFAv1 data found
    Iteration: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 129/129 [00:02<00:00, 45.35it/s]
    0 2.3439383506774902 2.3439383506774902 2.3439383506774902
    [[150, 1, '_wrapped_object.graph_pred_linear', array([  0, 277]), 4.0, 36.0]]
    0 1.8567776679992676 1.8567776679992676 1.8567776679992676
    [[150, 2, '_wrapped_object.graph_pred_linear', array([  0, 188]), -3.0, -35.0]]
    0 1.399310827255249 1.399310827255249 1.399310827255249
    [[150, 3, '_wrapped_object.graph_pred_linear', array([  0, 167]), -2.0, -34.0]]
    0 0.9056546688079834 0.9056546688079834 0.9056546688079834
    [[150, 4, '_wrapped_object.graph_pred_linear', array([  0, 295]), 3.0, 35.0]]
    0 0.5474755764007568 0.5474755764007568 0.5474755764007568
    [[150, 5, '_wrapped_object.graph_pred_linear', array([  0, 117]), -3.0, -35.0]]
    0 0.46581828594207764 0.46581828594207764 0.46581828594207764
    [[108, 6, '_wrapped_object.gnn_node.convs.3.mlp.3', array([266, 243]), 0.0, 32.0]]
    0 0.4379805326461792 0.4379805326461792 0.4379805326461792
    [[18, 7, '_wrapped_object.gnn_node.convs.0.mlp.0', array([322,   0]), -7.0, -39.0]]
    0 0.41478240489959717 0.41478240489959717 0.41478240489959717
    [[108, 8, '_wrapped_object.gnn_node.convs.3.mlp.3', array([144,  68]), 2.0, 34.0]]
    0 0.3999355733394623 0.3999355733394623 0.3999355733394623
    [[83, 9, '_wrapped_object.gnn_node.convs.2.mlp.3', array([91, 70]), -9.0, -41.0]]
    0 0.37580955028533936 0.37580955028533936 0.37580955028533936
    [[108, 10, '_wrapped_object.gnn_node.convs.3.mlp.3', array([144, 483]), -2.0, -34.0]]
    Iteration: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 129/129 [00:02<00:00, 47.33it/s]
    Current time: 03/06/2024, 13:26:47 Completed: 100.0 % Duration per experiment: 403.26 s ETA: 03/06/2024, 13:26:47
    Pre and post attack average metric, bitflips, failures 0.7062477065992006 0.4733859286583364 10.0 0
    End time 03/06/2024, 13:26:47

Our implementation of IBFA is based on the code of the original [Progressive Bitflip Attack](https://github.com/elliothe/Neural_Network_Weight_Attack)
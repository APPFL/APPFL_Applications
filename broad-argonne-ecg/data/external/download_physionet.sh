#!/bin/bash

wget -O WFDB_CPSC2018.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_CPSC2018.tar.gz/
wget -O WFDB_CPSC2018_2.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_CPSC2018_2.tar.gz/
wget -O WFDB_StPetersburg.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining//WFDB_StPetersburg.tar.gz/
wget -O WFDB_PTB.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_PTB.tar.gz/
wget -O WFDB_PTBXL.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_PTBXL.tar.gz/
wget -O WFDB_Ga.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_Ga.tar.gz/
wget -O WFDB_ChapmanShaoxing.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_ChapmanShaoxing.tar.gz/
wget -O WFDB_Ningbo.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_Ningbo.tar.gz/ \
wget https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/dx_mapping_scored.csv \
wget https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/dx_mapping_unscored.csv

ls *.tar.gz |xargs -n1 tar -xzf

# Defense against fake Cyber Threat Intelligence through Deep Learning
Code used for [IntelligentShield: Safeguarding Against Fake Cyber Threat Intelligence](http://).



## Updates
- May-27-2023: first released


## 1. Set-up
### environment requirements:
python = 3.7
```
pip install -r requirements.txt
```

## 2. Generate Dataset
prepare data folder, and choose from [opt 1 - use our dataset] or [opt 2 - generate your own dataset]
```
mkdir data
```

### Opt 1 - use our dataset: run following commands and jump to Step 3.
```
cp ./dataset/CTI_long.xlsx ./data/dataset_long.xlsx
```

### Opt 2 - generate your own dataset
download corpus dataset
```
cd data
git clone https://github.com/UMBC-Onramp/CyEnts-Cyber-Blog-Dataset.git
git clone https://github.com/Ebiquity/CASIE.git
```

generate short cti sample
```
cd ..
python generate_corpus.py --input CyEnts-Cyber-Blog-Dataset/Sentences/ --output UMBC_finetune.txt
```

prepare model folder
```
mkdir model
cd model
mkdir gpt2finetune
cd gpt2finetune
git clone https://github.com/nshepperd/gpt-2
```

Grant Colab read and execute access to the cloned folder.
```
chmod 755 -R ./gpt-2
```

Download the required GPT-2 model from the available four options, 124M, 355M, 774M, 1558M.
We use 355M
```
cd gpt-2
python download_model.py 355M
```

set python IO encoding to UTF-8
```
export PYTHONIOENCODING=UTF-8
```

Finetune gpt-2 model using CASIE_finetune.txt
```
PYTHONPATH=src ./train.py --dataset ../../../data/CASIE_finetune.txt --model_name 355M --batch_size 1 --memory_saving_gradients 2>&1 | tee casie_log1.txt
```

Save finetune model to 355M-v1
```
cd models
mkdir 355M-v1
cd ..
cp -r ./checkpoint/run1/*  ./models/355M-v1/
```

Copy generate_dataset_from_corpus.py and run
```
cp ../../../generate_dataset_from_corpus.py ./src/generate_dataset_from_corpus.py
python src/generate_dataset_from_corpus.py --top_k 40 --model_name 355M-v1 --input_dataset ../../../data/CASIE_finetune.txt --output_file ../../../data/dataset_long.xlsx
cd ../../..
```

## 3. Generate features from xlsx
### prepare resources
```
cd ./model
python -m spacy download en_core_web_lg
```

### Download GoogleNews-vectors-negative300.bin.gz file 
Download GoogleNews-vectors-negative300.bin.gz file into ./model from https://github.com/mmihaltz/word2vec-GoogleNews-vectors
Download word2vec_million_cybersecurity_docs_models.tar.gz into ./model from link on https://github.com/UMBC-ACCL/cybersecurity_embeddings, and unzip to ./model. (will get 3 files including 1million.word2vec.model in ./model)
### generate features
```
cd ..
python ./genarate_features_from_dataset.py --input dataset_long.xlsx --output dataset_long_with_feature.xlsx
```


## 4. Feature-based analyse
### feature analyse
```
python ./analyse_on_features.py --input dataset_long_with_feature.xlsx --function analyse
```

### feature selection - K Best
```
python ./analyse_on_features.py --input dataset_long_with_feature.xlsx --function feature_selection_k_best --k 2
```

### feature selection - Mutual information and maximal information coefficient (MIC)
```
python ./analyse_on_features.py --input dataset_long_with_feature.xlsx --function feature_selection_mic
```

### logistic regression with all features
```
python ./analyse_on_features.py --input dataset_long_with_feature.xlsx --function logistic_regression
```

### logistic regression with selected features
```
python ./analyse_on_features.py --input dataset_long_with_feature.xlsx --function logistic_regression --features cosine_similarity_sklearn_pd cosine_similarity_sklearn wmd_domain_pd wmd_google_pd wmd_cyber_pd
```

### random forest with selected features
```
python ./analyse_on_features.py --input dataset_long_with_feature.xlsx --function random_forest --features cosine_similarity_sklearn_pd cosine_similarity_sklearn wmd_domain_pd wmd_google_pd wmd_cyber_pd
```

## 4. Text-based analyse
### text analyse
```
python ./analyse_on_text.py --input dataset_long_with_feature.xlsx --function analyse
```

### Passive Aggressive Classifier
```
python ./analyse_on_text.py --input dataset_long_with_feature.xlsx --function passive_aggressive
```

### Logistic Regression Classifier
```
python ./analyse_on_text.py --input dataset_long_with_feature.xlsx --function logic_regression
```

### Decision tree Classifier
```
python ./analyse_on_text.py --input dataset_long_with_feature.xlsx --function decision_tree
```

### Random forest Classifier
```
python ./analyse_on_text.py --input dataset_long_with_feature.xlsx --function random_forest
```

### RoBERTa Classifier - pre-trained
```
cd model
wget https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-large.pt
cd ..

python ./analyse_on_text.py --input dataset_long_with_feature.xlsx --function roberta
```

### GLTR: Giant Language Model Test Room (using pretrained gpt-2 model) and Logistic Regression on the scores
Generate GLTR scores
```
python ./analyse_on_text.py --input dataset_long_with_feature.xlsx --output dataset_long_with_feature.xlsx --function gltr
```

Feature selection
```
python ./analyse_on_features.py --input dataset_long_with_feature.xlsx --function feature_selection_k_best --k 2 --features sentiment_topic_pd sentiment_content_pd jaccard_coef_pd sentence_avg_cosine_similarity_bert cosine_similarity_sklearn cosine_similarity_sklearn_pd cosine_similarity_spacy cosine_similarity_spacy_pd wmd_google_nonsplit_pd wmd_google_pd wmd_domain_pd wmd_cyber_nonsplit_pd wmd_cyber_pd gltr_gpt2_count_less_than_10_percentage gltr_gpt2_count_10_to_100_percentage gltr_gpt2_count_100_to_1000_percentage gltr_gpt2_count_more_than_1000_percentage
```

Logistic Regression with selected features
```
python ./analyse_on_features.py --input dataset_long_with_feature.xlsx --function logistic_regression --features gltr_gpt2_count_less_than_10_percentage gltr_gpt2_count_10_to_100_percentage gltr_gpt2_count_100_to_1000_percentage gltr_gpt2_count_more_than_1000_percentage
```

### Breydon Verryt-Reid - DeepLearning
```
pip uninstall transformers
pip install transformers==3.5.1

python ./analyse_on_text.py --input dataset_long_with_feature.xlsx --output dataset_long_with_feature.xlsx --function deep_learning
```

### ELMO
```
python ./analyse_on_text.py --input dataset_long_with_feature_debug.xlsx --function elmo
```

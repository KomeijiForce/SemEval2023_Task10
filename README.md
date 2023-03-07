# Chride Solution at SemEval2023_Task10

Pytorch implementation for the **Chride at SemEval-2023 Task 10: Fine-tuned DebertaV3 on Detection of Online Sexism with Hierarchical Loss** at SemEval2023 Task 10.


## Train & evaluate the model
To train and evaluate the model with a single GPU, try the following command:
```
python main.py \
    --config_file {config_file_path} \
    --data_file {data_file_path} \
```
You can run on different tasks by setting the target in the Config.json file.
- Task A: set "target": "sexist", "disable_neg_train": false, "disable_neg_eval": false.
- Task B: set "target": "category", "disable_neg_train": true, "disable_neg_eval": true.
- Task C: set "target": "vector", "disable_neg_train": true, "disable_neg_eval": true.


## Config.json
```
{
    "path": "microsoft/deberta-v3-large",
    "epochs": 20,
    "lr": 1e-5,
    "bottleneck_dim": 64,
    "batch_size": 8,
    "gradient_accumulation_steps": 2,
    "device": "cuda:0",
    "loss_weight": {"vector": 1.0, "category": 1.0, "sexist": 1.0},
    "target": "sexist",
    "disable_neg_train": false,
    "disable_neg_eval": false,
    "save_file": false,
    "labels": {
        "label_sexist":{
            "0":"not sexist",
            "1":"sexist"
        },
        "label_category":{
            "0":"none",
            "1":"1. threats, plans to harm and incitement",
            "2":"2. derogation",
            "3":"3. animosity",
            "4":"4. prejudiced discussions"
        },
        "label_vector":{
            "0":"none",
            "1.1":"1.1 threats of harm",
            "1.2":"1.2 incitement and encouragement of harm",
            "2.1":"2.1 descriptive attacks",
            "2.2":"2.2 aggressive and emotive attacks",
            "2.3":"2.3 dehumanising attacks & overt sexual objectification",
            "3.1":"3.1 casual use of gendered slurs, profanities, and insults",
            "3.2":"3.2 immutable gender differences and gender stereotypes",
            "3.3":"3.3 backhanded gendered compliments",
            "3.4":"3.4 condescending explanations or unwelcome advice",
            "4.1":"4.1 supporting mistreatment of individual women",
            "4.2":"4.2 supporting systemic discrimination against women as a group"
        }
    }
}
```

## Performance


|  | Task A (Dev) | Task A (Test) | Task B (Dev) | Task B (Test) | Task C (Dev) | Task C (Test) |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| BERT_large | 82.74 | 82.63 | 61.16 | 57.48 | 36.65 |   36.17 |
| RoBERTa_large | 84.29 | 83.58 | 61.09 | 57.88 | 42.28 | 37.57 |
| BERTweet_large | 84.97 | 84.64| 0.130 | 0.341 | 46.90 | 39.22 |
| DeBERTaV3_large | 85.48 | 85.67 | 70.39 | 66.36 | 0.597 | 45.61 | 38.25 |

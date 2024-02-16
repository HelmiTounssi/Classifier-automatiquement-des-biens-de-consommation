import json
import os

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from multimodal_training import from_pretrained, set_seed
args = {
        "batch_size": 16,
        "num_train_epochs": 1,
        "learning_rate": 1.0e-5,
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "max_seq_length": 64,
        "text_field": "text",
        "label_field": "label",
        "image_path_field": "img_path",
    }


def main():
   
    home_folder = "I:\\openclassroom\\projet-6-final\\monprojet\\Flipkart\\"
    data_folder = home_folder 
    image_folder = data_folder + "images\\"
    results_folder = home_folder + "results\\"
    trained_models_folder = home_folder + "trained_models\\"
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(trained_models_folder, exist_ok=True)
    
    os.makedirs(results_folder, exist_ok=True)
    
    df_X_val= pd.read_csv(home_folder + 'test.csv')
    df_y_val= pd.read_csv(home_folder + 'test_labels.csv')
  
    df_X_val['text']= df_X_val['description']
   
    df_X_val['label']= df_y_val['label']
    
    df_X_val[args["image_path_field"]] = df_X_val[args["image_path_field"]].apply(
        lambda x: image_folder + x
    )
    seed_val = 0
    set_seed(seed_val)
    
    # load pretrained bert model and predict on test set

    bert_folder = trained_models_folder + "bert\\"
    with open(bert_folder + "parameters.json", "r") as f:
        bert_args = json.load(f)

    bert_classifier = from_pretrained(bert_folder)
    bert_predictions = bert_classifier.predict(df_X_val.copy(), bert_args)
    bert_class_report = classification_report(
        df_y_val[bert_args.get("label_field")], bert_predictions, output_dict=True
    )
    print(bert_class_report)
    print("BERT Accuracy:", bert_class_report["accuracy"])

    # load pretrained bert-resnet model and predict on test set
    set_seed(seed_val)
    bert_resnet_folder = trained_models_folder + "bert_resnet\\"
    with open(bert_resnet_folder + "parameters.json", "r") as f:
        bert_resnet_args = json.load(f)

    bert_resnet_classifier = from_pretrained(bert_resnet_folder)
    bert_resnet_predictions = bert_resnet_classifier.predict(
        df_X_val.copy(), bert_resnet_args
    )
    bert_resnet_class_report = classification_report(
        df_y_val[bert_resnet_args.get("label_field")],
        bert_resnet_predictions,
        output_dict=True,
    )
    print(bert_resnet_class_report)
    print("BERT-ResNet Accuracy:", bert_resnet_class_report["accuracy"])

    # load pretrained ALBEF model and predict on test set
    set_seed(seed_val)
    albef_folder = trained_models_folder + "albef\\"
    with open(albef_folder + "parameters.json", "r") as f:
        albef_args = json.load(f)

    albef_classifier = from_pretrained(albef_folder)
    albef_predictions = albef_classifier.predict(df_X_val, albef_args)
    albef_class_report = classification_report(
        df_y_val[albef_args.get("label_field")], albef_predictions, output_dict=True
    )
    print(albef_class_report)
    print("ALBEF Accuracy:", albef_class_report["accuracy"])


if __name__ == "__main__":
    main()

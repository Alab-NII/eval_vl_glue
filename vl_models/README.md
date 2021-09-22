# eval\_vl\_glue/vl\_models

Directory for pre-trained and fine-tuned models.

- **convert.py** converts the original weights of the volta models to the transformers_volta models.
- **batch_convert.sh** is a batch version of convert.py to make the five V\&L models.
- **test_obj36.tsv** contains some image features extracted by the original BUTD detector. Those features are used to make the default image features, features used when no image input is provided, for the V\&L models.
- **pretrained** is a directory to place the pre-trained models.
- **finetuned** is a directory to output the fine-tuning results.

## Weights for the transformers_volta models

We provide two methods to prepared the weights: **conversion** and **download**.  
The created models can be used from the transfomers_volta by specifing their path.

### Conversion

We describe the ctrl_vilbert case as an example.

1. download the original weight from the link **ViLBERT (CTRL)** in [e-bug/volta/MODELS.md](https://github.com/e-bug/volta/blob/main/MODELS.md) into the download/volta\_weights directory, with the name *ctrl_vilbert*.
2. run convert.py from the repository root directory.

    ```
    python convert.py --config download/volta_config/ctrl_vilbert_base.json --weight download/volta_weights/ctrl_vilbert
    ```

This will make a ctrl\_vilbert (directory) in vl\_models/pretrained.

We prepared batch\_convert.sh to make the five models used our experiments (ctrl_visual_bert, ctrl_uniter, ctrl_vl_bert, ctrl_lxmert and ctrl_vilbert).
Run this batch script from the repository root directory after downloading and putting the corresponding weights.

### Download

Alternatively, you can download model files we made.
Here is the url list.  
To use those models, after downloading, unzip them and put the unzipped model directories into vl\_models/pretrained.

We made those model files from the [e-bug/volta](https://github.com/e-bug/volta) work.
If you use these weights, do not forget to cite their work appropriately.

**Controlled models:**

| file | url |
| ---- | --- |
| ctrl_visual_bert.zip | https://iki-my.sharepoint.com/personal/ikitaichi_iki_onmicrosoft_com/_layouts/15/download.aspx?share=EZoveXHRYcRCm-QRjwjq4HIBLqxq8A_2YdoJdEH0IcvLAQ |
| ctrl_uniter.zip | https://iki-my.sharepoint.com/personal/ikitaichi_iki_onmicrosoft_com/_layouts/15/download.aspx?share=ERkcnp3Kp-pLiM6OQuJlgMQBfLs02lpjbg2lUCRkWSlrCg |
| ctrl_vl_bert.zip | https://iki-my.sharepoint.com/personal/ikitaichi_iki_onmicrosoft_com/_layouts/15/download.aspx?share=EYzN_zgbp4BBi971d7484G8BQYWpS7qiaQ4azIiQnG4lFw |
| ctrl_lxmert.zip | https://iki-my.sharepoint.com/personal/ikitaichi_iki_onmicrosoft_com/_layouts/15/download.aspx?share=ETkGAzIBfwRDq8t-O0l_t_8B_diFfc0qvXHdULvIUlixVQ |
| ctrl_vilbert.zip | https://iki-my.sharepoint.com/personal/ikitaichi_iki_onmicrosoft_com/_layouts/15/download.aspx?share=EXCWGdS4Pc1GqU2uvnLD4E4BHzD38tqMQnoLITsaKzqPMg |

**Reinitialized models (initialized randomly and transferred some weights from BERT-base-uncased):**

| file | url |
| ---- | --- |
| ctrl_visual_bert_reinit.zip | https://iki-my.sharepoint.com/personal/ikitaichi_iki_onmicrosoft_com/_layouts/15/download.aspx?share=ETxdhUtDvE9IsnlLt29w7pIBqPDDZ7j7PwJGwqkzzQZWEA |
| ctrl_uniter_reinit.zip | https://iki-my.sharepoint.com/personal/ikitaichi_iki_onmicrosoft_com/_layouts/15/download.aspx?share=EYEd1HgIk91ClpA6c9Ne9REBZXg0sZlEcnoqKXrf3VtIxg |
| ctrl_vl_bert_reinit.zip | https://iki-my.sharepoint.com/personal/ikitaichi_iki_onmicrosoft_com/_layouts/15/download.aspx?share=EWHfsOSW06lGq24vErTxNywBEXM_xz-2RrEUpdKaYDwB8g |
| ctrl_lxmert_reinit.zip | https://iki-my.sharepoint.com/personal/ikitaichi_iki_onmicrosoft_com/_layouts/15/download.aspx?share=EVj5ZaadBdBIthVkPQtQ5nQBlIVempj3paAfb6VOAm7_0w |
| ctrl_vilbert_reinit.zip | https://iki-my.sharepoint.com/personal/ikitaichi_iki_onmicrosoft_com/_layouts/15/download.aspx?share=EWPY476hDwJMmzvcIzbFoO8BYinPJW3Lev9FpQIP9nJt9g |

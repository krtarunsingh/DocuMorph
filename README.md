
# Legal Document Paraphraser

This repository contains a tool for paraphrasing legal documents. It uses a fine-tuned BERT model to generate paraphrases.

## Dataset
The dataset used for fine-tuning can be found [here](https://www.kaggle.com/datasets/vladimirvorobevv/chatgpt-paraphrases).

## Setup and Installation
### Step 1: Install Necessary Packages
Install the required packages by running the following command in your terminal:
```
pip install transformers
pip install torch
pip install streamlit
```

### Step 2: Clone the Repository
Clone this repository to your local machine and navigate to the repository folder.

### Step 3: Download the Dataset
Download the dataset from the link provided above and place it in the `data` folder with the filename `dataset.csv`.

### Step 4: Fine-Tune the Model
Run the `fine_tune.py` script to fine-tune the BERT model on the dataset. You can do this by executing the following command in your terminal:
```
python fine_tune.py
```

### Step 5: Run the Streamlit App
Once the model has been fine-tuned, you can run the Streamlit app using the following command:
```
streamlit run app.py
```

## Training Parameters Explanation
The training script (`fine_tune.py`) utilizes several parameters for fine-tuning the model, explained as follows:
1. `output_dir`: The directory where the output model files will be saved.
2. `overwrite_output_dir`: If set to True, it will overwrite the output directory if it already exists.
3. `num_train_epochs`: The number of training epochs. Each epoch is a full pass over the training data.
4. `per_device_train_batch_size`: The batch size per training step.
5. `save_steps`: The number of training steps after which the model is saved.
6. `save_total_limit`: The maximum number of saved model checkpoints.
7. `evaluation_strategy`: The strategy to use for evaluation of the model. Setting it to "epoch" means it will evaluate at the end of each epoch.

## Usage
Open the Streamlit app in your web browser. You will see a text box where you can input a legal paragraph that you want to paraphrase. Click the "Paraphrase" button to generate paraphrases of the input text.

## Sample Inputs and Outputs
### Input
"Under the provisions of Section 54 of the Capital Gains Act, 1961, any long-term capital gains arising from the sale of a residential property are subject to tax in India. However, if the capital gains from the sale are reinvested in another residential property within a period of two years, the taxpayer can claim an exemption on the capital gains tax. This provision is aimed at promoting investment in the real estate sector."

### Output
1. "As per Section 54 of the Capital Gains Act, 1961, the tax is levied on any long-term capital gains incurred from selling a residential property in India. Nonetheless, an exemption on the capital gains tax can be claimed by the taxpayer if the proceeds from the sale are reinvested in a new residential property within two years. This clause encourages investments in the real estate industry."
2. "According to the Capital Gains Act of 1961, specifically Section 54, long-term capital gains resulting from the disposal of a residential asset are taxable in India. But, the taxpayer is entitled to a tax exemption if the capital gains are redirected into the purchase of another residential property within a span of two years. The intention behind this provision is to boost investments in the property market."
3. "In line with Section 54 of the Capital Gains Act, 1961, long-term capital gains generated from the transaction of a residential property are liable to taxation in India. Yet, the capital gains tax can be exempted if the sales gains are invested into another residential property in a timeframe of two years. This rule is designed to foster investments in the housing sector."
4. "Section 54 of the Capital Gains Act, 1961 stipulates that long-term capital gains stemming from the sale of residential real estate are taxable in India. Conversely, a tax exemption on the capital gains can be availed if the sale proceeds are invested in another residential property within a two-year window. This provision is orchestrated to stimulate investments in real estate."
5. "Pursuant to Section 54 of the Capital Gains Act, 1961, any long-term capital gains accrued from the sale of residential property are subject to taxation in India. However, a waiver on the capital gains tax can be secured if the gains from the sale are channelled into another residential property over two years. This provision serves to escalate investments in the real estate domain."

## No License

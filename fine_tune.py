
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSeq2SeqLM.from_pretrained("bert-base-uncased")

# Load the dataset
data = pd.read_csv('data/dataset.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)

# Preprocess the data
def preprocess_function(examples):
    inputs = examples['text'].tolist()
    targets = examples['paraphrases'].apply(lambda x: eval(x)).tolist() # Assuming 'paraphrases' is a string representation of a list
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_data.apply(preprocess_function, axis=1)
test_dataset = test_data.apply(preprocess_function, axis=1)

# Fine-tune the model
training_args = Seq2SeqTrainingArguments(
    output_dir="./LegalBert",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./LegalBert")
tokenizer.save_pretrained("./LegalBert")

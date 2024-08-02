from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model,TaskType
import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task = "qnli"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
rank=4
d_model=768

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


#--------------- PREPROCESS
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)


num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2


#------------------------------------------------------------------------------------------
#------------------TASK1
model_dir = 'distilBert_lora_cola'
model_task1 = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True).to(device)
for param in model_task1.parameters():
    param.requires_grad = False
#------------------TASK2
model_dir = 'distilBert_lora_mnli-mm'
model_task2 = AutoModelForSequenceClassification.from_pretrained(model_dir,num_labels=3, output_hidden_states=True).to(device)
for param in model_task2.parameters():
    param.requires_grad = False
#------------------TASK3
model_dir = 'distilBert_lora_mnli'
model_task3 = AutoModelForSequenceClassification.from_pretrained(model_dir,num_labels=3, output_hidden_states=True).to(device)
for param in model_task3.parameters():
    param.requires_grad = False
#------------------TASK4
model_dir = 'distilBert_lora_mrpc'
model_task4 = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True).to(device)
for param in model_task4.parameters():
    param.requires_grad = False
#------------------TASK5
model_dir = 'distilBert_lora_qnli'
model_task5 = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True).to(device)
for param in model_task5.parameters():
    param.requires_grad = False
#------------------TASK6
model_dir = 'distilBert_lora_qqp'
model_task6 = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True).to(device)
for param in model_task6.parameters():
    param.requires_grad = False
#------------------TASK7
model_dir = 'distilBert_lora_rte'
model_task7 = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True).to(device)
for param in model_task7.parameters():
    param.requires_grad = False
#------------------TASK8
model_dir = 'distilBert_lora_sst2'
model_task8 = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True).to(device)
for param in model_task8.parameters():
    param.requires_grad = False
#------------------TASK9
model_dir = 'distilBert_lora_wnli'
model_task9 = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True).to(device)
for param in model_task9.parameters():
    param.requires_grad = False


#-----------------------------------------------------------------------------------------------
#----------------------------------------combine model -----------------------------------------
class CombinedModel(nn.Module):
    def __init__(self, base_model, model1, model2, model3,model4, model5, model6,model7, model8, model9, num_labels):
        super(CombinedModel, self).__init__()
        self.base_model = base_model
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6
        self.model7 = model7
        self.model8 = model8
        self.model9 = model9
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4)
        self.feed_forward = nn.Linear(d_model, d_model)
        
        self.classifier = nn.Linear(768, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()  

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs1 = self.model1(input_ids, attention_mask, token_type_ids).hidden_states[-1][:, 0, :]
        outputs2 = self.model2(input_ids, attention_mask, token_type_ids).hidden_states[-1][:, 0, :]
        outputs3 = self.model3(input_ids, attention_mask, token_type_ids).hidden_states[-1][:, 0, :]
        outputs4 = self.model4(input_ids, attention_mask, token_type_ids).hidden_states[-1][:, 0, :]
        outputs5 = self.model5(input_ids, attention_mask, token_type_ids).hidden_states[-1][:, 0, :]
        outputs6 = self.model6(input_ids, attention_mask, token_type_ids).hidden_states[-1][:, 0, :]
        outputs7 = self.model7(input_ids, attention_mask, token_type_ids).hidden_states[-1][:, 0, :]
        outputs8 = self.model8(input_ids, attention_mask, token_type_ids).hidden_states[-1][:, 0, :]
        outputs9 = self.model9(input_ids, attention_mask, token_type_ids).hidden_states[-1][:, 0, :]

         # Project logits to embedding dimension
        # proj_outputs1 = self.projection(outputs1)
        # proj_outputs2 = self.projection(outputs2)
        # proj_outputs3 = self.projection(outputs3)

        # combined_outputs = torch.stack([proj_outputs1, proj_outputs2, proj_outputs3], dim=0)
        combined_outputs = torch.stack((outputs1, outputs2, outputs3,outputs4, outputs5, outputs6,outputs7, outputs8, outputs9), dim=1)
        attn_output, _ = self.attention(combined_outputs, combined_outputs, combined_outputs)
     

        # Aggregate outputs
        agg_output = torch.sum(attn_output, dim=1) 
       

        # Pass through feed-forward layer for task 4
        outputs4 = self.base_model(input_ids=input_ids, attention_mask=attention_mask).hidden_states[-1][:, 0, :]

        outputs_task4 = self.feed_forward(outputs4)


        final_output=outputs_task4 + agg_output

        
        logits = self.classifier(final_output)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        
        return logits

model_checkpoint = "distilbert-base-uncased"
base_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels, output_hidden_states=True).to(device)
for param in base_model.parameters():
    param.requires_grad = False

combined_model = CombinedModel(base_model, model_task1,model_task2, model_task3,
                               model_task4,model_task5, model_task6,
                               model_task7,model_task8, model_task9,
                                num_labels).to(device)



#--------------------TRAINER
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-combine_finetuned-{task}",
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    # push_to_hub=True,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    combined_model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
#--------------------------SAVE MODEL
# combined_model.save_pretrained(f'combine_{task}')
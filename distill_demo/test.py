import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, AutoConfig
import pandas as pd

# === 1) Load your data ===
df = pd.read_csv('classified_texts.csv')
texts = df['inner_text'].tolist()
labels = df['electric_car'].map({'Y': 1, 'N': 0}).tolist()

# === 2) Load tokenizer and base BERT ===
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = AutoConfig.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased', config=config)

# === 3) Make dataset ===
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

dataset = TextDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# === 4) Build simple classification head ===
class Classifier(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # shape [batch, hidden]
        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits

model = Classifier(bert)

# === 5) Train ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

epochs = 3

for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} loss: {loss.item():.4f}")

# === 6) Save your fine-tuned model for sharding later ===
torch.save(model.state_dict(), 'fine_tuned_bert_classifier.pt')

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

import pprint

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler

import os
from IPython.display import display, clear_output
from matplotlib import pyplot as plt
from matplotlib.pyplot import show, plot
from IPython.display import display
import ipywidgets as widgets
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ### search the script for .todevice() so uncomment when wanting to use GPU
print(f"Using device: {device}")

text = Path('tiny-shakespeare.txt').read_text()
#print(text[0:1000])

### Class Definition
class CharTokenizer:
    def __init__(self, vocabulary):
        self.token_id_for_char = {char: token_id for token_id, char in enumerate(vocabulary)}
        self.char_for_token_id = {token_id: char for token_id, char in enumerate(vocabulary)}

    @staticmethod #newly added
    def train_from_text(text):
        vocabulary = set(text)
        return CharTokenizer(sorted(list(vocabulary)))

    def encode(self, text):
        token_ids = []
        for char in text:
            token_ids.append(self.token_id_for_char[char])
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode(self, token_ids):
        chars = []
        for token_id in token_ids.tolist():
            chars.append(self.char_for_token_id[token_id])
        return "".join(chars)
    
    def vocabulary_size(self):
        return len(self.token_id_for_char)
    
tokenizer = CharTokenizer.train_from_text(text)
# a = tokenizer.encode("Hello World")
# print(a)
# b = tokenizer.decode(a)
# print(b)

# print(f"Vocabulary size: {tokenizer.vocabulary_size()}")

# pp = pprint.PrettyPrinter(depth=4)
# pp.pprint(tokenizer.char_for_token_id)
# pp.pprint(tokenizer.token_id_for_char)


## DATALOADER
class TokenIdsDataset(Dataset):
  def __init__(self, data, block_size):
    self.data = data
    self.block_size = block_size
    
  def __len__(self):
    return len(self.data)-self.block_size

  def __getitem__(self, pos):
    # TODO: Check if the input position is valid
    assert pos < len(self.data)-self.block_size
    a = self.data[pos:pos + self.block_size]
    b = self.data[pos+1:pos+1 + self.block_size]
    return a,b

# Step 2 - Tokenize the Text
#tokenized = tokenizer.encode(text)
# # TODO: Encode text using the tokenizer
# dataset = TokenIdsDataset(tokenized, block_size=64)
# # Create "TokenIdsDataset" with the tokenized text, and block_size=64

# # Step 3 - Retrieve the First Item from the Dataset
# a, b = dataset[0]
# print(a,b)
# print(tokenizer.decode(b))
# # TODO: Get the first item from the dataset
# # Decode "x" using tokenizer.decode

# # RandomSampler allows to read random items from a datasset
# sampler = RandomSampler(dataset, replacement=True)
# # Dataloader will laod two random samplers using the sampler
# dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

# # Step 4 - Use a DataLoader
# # TODO: Get a single batch from the "dataloader"
# a,b = next(iter(dataloader))
# # For this call the `iter` function, and pass DataLoader instance to it. This will create an iterator
# # Then call the `next` function and pass the iterator to it to get the first training batch

# # TODO: Decode input item
# print(a.shape)
# print(tokenizer.decode(a[0]))

# # TODO: Decode target item
# print(tokenizer.decode(b[0]))

### Attention Block
config = {
   "vocabulary_size": tokenizer.vocabulary_size(),
   "context_size": 256,
   "d_embed": 768,
   "heads_num": 12,
   "layers_num": 10,
   "dropout_rate": 0.1,
   "use_bias": False,
}
config["head_size"] = config["d_embed"] // config["heads_num"]

class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Q_weights = nn.Linear(config["d_embed"], config["head_size"], config["use_bias"])
        self.K_weights = nn.Linear(config["d_embed"], config["head_size"], config["use_bias"])
        self.V_weights = nn.Linear(config["d_embed"], config["head_size"], config["use_bias"])

        self.dropout = nn.Dropout(config["dropout_rate"])

        casual_attention_mask = torch.tril(torch.ones(config["context_size"], config["context_size"]))
        self.register_buffer('casual_attention_mask', casual_attention_mask)

    def forward(self,input): # (B, C, embedding dim) [batch size, size of context window, embedding vector]
        batch_size, tokens_num, d_embed = input.shape
        Q = self.Q_weights(input)
        K = self.K_weights(input)
        V = self.V_weights(input)

        attention_scores = Q @ K.transpose(1,2) # (B,C,C)
        attention_scores = attention_scores.masked_fill(
         self.casual_attention_mask[:tokens_num, :tokens_num] == 0,
         -torch.inf
        )
        attention_scores = attention_scores / (K.shape[-1] **.5)
        # print(attention_scores.shape,"Pre Softmax")
        # print(attention_scores)
        attention_scores = torch.softmax(attention_scores, dim=-1) #DID YOU DO THIS
        # attention_scores_wrong = torch.softmax(attention_scores, dim=1)
        # print(attention_scores.shape,"Post Softmax")
        # print(attention_scores)
        # print(attention_scores_wrong.shape,"Incorrect Softmax")
        # print(attention_scores_wrong)
        attention_scores = self.dropout(attention_scores)

        return attention_scores @ V
    
# input = torch.rand(8, config["context_size"], config["d_embed"])
# ah = AttentionHead(config)
# output = ah(input)
# print(output.shape)

class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    heads_list = [AttentionHead(config) for _ in range(config["heads_num"])]
    self.heads = nn.ModuleList(heads_list)
    self.linear = nn.Linear(config["d_embed"], config["d_embed"])
    self.dropout = nn.Dropout(config["dropout_rate"])

  def forward(self, input):
    heads_outputs = [head(input) for head in self.heads]
    scores_change = torch.cat(heads_outputs, dim=-1)
    scores_change = self.linear(scores_change)
    return self.dropout(scores_change)
  
# mha = MultiHeadAttention(config)
# output = mha(input)
# print(output.shape)  # Expected output: torch.Size([8, 256, 768])

class FeedForward(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.linear_layers = nn.Sequential(
        nn.Linear(config["d_embed"], config["d_embed"] * 4),
        nn.GELU(),
        nn.Linear(config["d_embed"] * 4, config["d_embed"]),
        nn.Dropout(config["dropout_rate"])
    )

  def forward(self, input):
    return self.linear_layers(input)
  
# ff = FeedForward(config)
# input = torch.rand(8, config["context_size"], config["d_embed"])
# output = ff(input)
# print(output.shape)  # Expected: torch.Size([8, 256, 768])

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.multi_head = MultiHeadAttention(config)
    self.layer_norm_1 = nn.LayerNorm(config["d_embed"])
    self.feed_forward = FeedForward(config)
    self.layer_norm_2 = nn.LayerNorm(config["d_embed"])

  def forward(self, input):
    residual = input
    x = self.multi_head(self.layer_norm_1(input))
    x = x + residual

    residual = x
    x = self.feed_forward(self.layer_norm_2(x))
    return x + residual
  
# b = Block(config)
# input = torch.rand(8, config["context_size"], config["d_embed"])
# output = b(input)
# print(output.shape)  # Expected: torch.Size([8, 256, 768])

class DemoGPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.token_embedding_layer = nn.Embedding(config["vocabulary_size"], config["d_embed"])
    self.positional_embedding_layer = nn.Embedding(config["context_size"], config["d_embed"])
    blocks = [Block(config) for _ in range(config["layers_num"])]
    self.layers = nn.Sequential(*blocks)
    self.layer_norm = nn.LayerNorm(config["d_embed"])
    self.unembedding = nn.Linear(config["d_embed"], config["vocabulary_size"], bias=False)

  def forward(self, token_ids):
    batch_size, tokens_num = token_ids.shape
    x = self.token_embedding_layer(token_ids)
    sequence = torch.arange(tokens_num, device=device)
    x = x + self.positional_embedding_layer(sequence)
    x = self.layers(x)
    x = self.layer_norm(x)
    x = self.unembedding(x)
    return x

### LOAD CHECKPOINT
checkpoint = torch.load("./checkpoint_NLP_Shakespeare.pth")

### VIEW UNTRAINED MODEL
model = DemoGPT(config).to(device)
#print("Our model: \n\n", model, '\n')
#print("The state dict keys: \n\n", model.state_dict().keys())
for k,v in model.state_dict().items():
  if k == "layer_norm.weight":
    print("Key =\n", k, "Value =\n", v, "\n")

### CONFIRM TRAINED MODEL LOADED
model.load_state_dict(checkpoint['state_dict'])
#print("Our model: \n\n", model, '\n')
#print("The state dict keys: \n\n", model.state_dict().keys())
for k,v in model.state_dict().items():
  if k == "layer_norm.weight":
    print(k, "Key\n", v, "Value\n\n")

# output = model(tokenizer.encode("Hi").unsqueeze(dim=0)).to(device))
# print(output.shape)  # Expected: torch.Size([1, 2, 65])

def generate(model, prompt_ids, max_tokens):
    output_ids = prompt_ids
    for _ in range(max_tokens):
      if output_ids.shape[1] >= config["context_size"]:
        break
      with torch.no_grad():
        logits = model(output_ids)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      next_token_id = torch.multinomial(probs, num_samples=1)
      output_ids = torch.cat([output_ids, next_token_id], dim=-1)
    return output_ids

def generate_with_prompt(model, tokenizer, prompt, max_tokens=100):
  model.eval()
  prompt = tokenizer.encode(prompt).unsqueeze(dim=0).to(device)
  return tokenizer.decode(generate(model, prompt, max_tokens=max_tokens)[0])

### Train the model

batch_size = 16
train_iterations = 5000
evaluation_interval = 50
learning_rate = .0001
train_split = 0.9

## Step 1 - Split Data into Training and Validation Dataset
tokenized_text = tokenizer.encode(text).to(device)
# TODO: Get number of tokens in the training dataset. Should be train_split * number_of_tokens
split_tokens = int(train_split * len(tokenized_text)) #int() makes it a whole number by rounding the float output

# TODO: Split data into training and validation datasets
train_data = tokenized_text[:split_tokens]
validation_data = tokenized_text[split_tokens:]


train_dataset = TokenIdsDataset(train_data, config["context_size"])
train_sampler = RandomSampler(train_dataset, num_samples=batch_size * train_iterations, replacement=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

## Step 2 - Create Validation Dataset
# TODO: Create a validation dataset from the validation data
validation_dataset = TokenIdsDataset(validation_data, config["context_size"])
validation_sampler = RandomSampler(validation_dataset, replacement=True) #why is there no num_samples on this one?
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler)

# Step 4 - Calculate Validation Loss
@torch.no_grad()
def calculate_validation_loss(model, batches_num):
    model.eval()
    total_loss = 0

    # TODO: Create an iterator for the validation data loader
    validation_iter = iter(validation_dataloader)
    for _ in range(batches_num):
        input, targets = next(validation_iter)
        logits = model(input)

        # TODO: Use the "view" method to convert logits and targets so we could use the "cross_entropy" function```python
        logits_view = logits.view(batch_size * config["context_size"], config["vocabulary_size"])
        targets_view = targets.view(batch_size * config["context_size"])

        # TODO: calculate cross entropy using logits and target data
        loss = F.cross_entropy(logits_view, targets_view)

        # TODO: Add loss to the "total_loss" variable
        total_loss += loss.item()
        # Note: you would need to use the "item()" method to convert a tensor to a number

    average_loss = total_loss / batches_num

    return average_loss

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
train_line, = ax.plot([], [], label='Training Loss')
val_line, = ax.plot([], [], label='Validation Loss')
ax.legend()
ax.set_xlabel('Step')
ax.set_ylabel('Loss')

def update_plot(train_line, val_line, train_steps, train_losses, val_steps, val_losses):
    train_line.set_data(train_steps, train_losses)
    val_line.set_data(val_steps, val_losses)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.01)

# Set up lists to store losses for plotting
train_losses = []
train_steps = []
eval_losses = []
eval_steps = []

for step_num, sample in enumerate(train_dataloader):

  model.train()
  input, targets = sample
  logits = model(input)

  logits_view = logits.view(batch_size * config["context_size"], config["vocabulary_size"])
  targets_view = targets.view(batch_size * config["context_size"])
  
  #logps = model.forward(input)
  #logps_view = logps.view(batch_size * config["context_size"], config["vocabulary_size"])
  #loss = F.cross_entropy(logps_view, targets_view)
  loss = F.cross_entropy(logits_view, targets_view)

  # Backward propagation
  loss.backward()
  # Update model parameters
  optimizer.step()
  # Set to None to reduce memory usage
  optimizer.zero_grad(set_to_none=True)

  train_losses.append(loss.item())
  train_steps.append(step_num)

  print(f"Step {step_num}. Loss {loss.item():.3f}")

  if step_num % evaluation_interval == 0 and step_num != 0:
    print("Demo GPT:\n" + generate_with_prompt(model, tokenizer, "\n"))

    validation_loss = calculate_validation_loss(model, batches_num=50)
    eval_losses.append(validation_loss)
    eval_steps.append(step_num)
    print(f"Step {step_num}. Validation loss: {validation_loss:.3f}")

    ## View it
    #print("Our model: \n\n", model, '\n')
    #print("The state dict keys: \n\n", model.state_dict().keys())

    # Save it
    checkpoint = {'state_dict': model.state_dict()}
    torch.save(checkpoint,'checkpoint_NLP_Shakespeare.pth')

  #update_plot(train_line, val_line, train_steps, train_losses, eval_steps, eval_losses)


# plt.ioff()
# plt.show()
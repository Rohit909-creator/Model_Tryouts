import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
from datasets import load_dataset

# torch.set_default_device('cuda')

# Your HRM model (paste your simplified model here)
from model2 import SimpleHierarchicalReasoningModel

# sudoku_dataset = TensorDataset(train_data[:train_data.shape[0]-1000], label_data[:train_data.shape[0]-1000])
# test_dataset = TensorDataset(train_data[train_data.shape[0]-1000:], label_data[train_data.shape[0]-1000:])


class SudokuDataset(Dataset):
    def __init__(self, split='train'):
        self.tokenizer = {char: i for i, char in enumerate(".123456789")}
        
        # self.train_data = torch.load("Data.pt")[:3831994-2000].to(torch.int16)
        # self.label_data = torch.load("Label_Data.pt")[:3831994-2000].to(torch.int16)
        
        # self.train_data = torch.load("./HRM/Data.pt", map_location='cuda')[:3831994-2000].long()
        # self.label_data = torch.load("./HRM/Label_Data.pt", map_location='cuda')[:3831994-2000].long()
        
        self.train_data = torch.load("./HRM/Data.pt", map_location='cuda')[:2000].long()
        self.label_data = torch.load("./HRM/Label_Data.pt", map_location='cuda')[:2000].long()
        
        # # Load tokenized questions
        # with open(data_file, 'r') as f:
        #     self.questions = [json.loads(line) for line in f]
        
        # # Get answers from original dataset
        # self.answers = []
        # for i in range(len(ds[split])):
        #     tokenized_answer = [self.tokenizer[char] for char in ds[split][i]['answer']]
        #     self.answers.append(tokenized_answer)
    

    
    def __len__(self):
        return self.train_data.shape[0]
    
    def __getitem__(self, idx):
        return {
            'inputs': self.train_data[idx],
            'targets': self.label_data[idx],
            'puzzle_identifiers': torch.tensor([0], dtype=torch.long)  # Single puzzle type for now
        }
        

class SudokuDataset2(Dataset):
    def __init__(self, split='train'):
        self.tokenizer = {char: i for i, char in enumerate(".123456789")}
        
        # self.train_data = torch.load("Data.pt")[3831994-2000:].to(torch.int16)
        # self.label_data = torch.load("Label_Data.pt")[3831994-2000:].to(torch.int16)
        
        # self.train_data = torch.load("./HRM/Data.pt", map_location='cuda')[3831994-2000:].long()
        # self.label_data = torch.load("./HRM/Label_Data.pt", map_location='cuda')[3831994-2000:].long()
        
        self.train_data = torch.load("./HRM/Data.pt", map_location='cuda')[2000:4000].long()
        self.label_data = torch.load("./HRM/Label_Data.pt", map_location='cuda')[2000:4000].long()
        # # Load tokenized questions
        # with open(data_file, 'r') as f:
        #     self.questions = [json.loads(line) for line in f]
        
        # # Get answers from original dataset
        # self.answers = []
        # for i in range(len(ds[split])):
        #     tokenized_answer = [self.tokenizer[char] for char in ds[split][i]['answer']]
        #     self.answers.append(tokenized_answer)
    

    
    def __len__(self):
        return self.train_data.shape[0]
    
    def __getitem__(self, idx):
        return {
            'inputs': self.train_data[idx],
            'targets': self.label_data[idx],
            'puzzle_identifiers': torch.tensor([0], dtype=torch.long)  # This should be scalar or match batch
        }

class HRMLightning(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        
        # HRM config
        BATCH_SIZE = 8  # Define batch size here
        config = {
            'batch_size': BATCH_SIZE,
            'seq_len': 81,  # 9x9 sudoku
            'vocab_size': 10,  # 0-9 digits
            'num_puzzle_identifiers': 1,
            'hidden_size': 256,
            'num_heads': 8,
            'H_layers': 2,
            'L_layers': 2,
            'H_cycles': 2,
            'L_cycles': 4,
            'halt_max_steps': 5
        }
        
        self.model = SimpleHierarchicalReasoningModel(
            vocab_size=10,
            seq_len=81,      # Smaller for demo
            hidden_size=256, # Smaller for demo
            num_heads=4,
            high_layers=2,
            low_layers=3,
            high_cycles=2,
            low_cycles=2,
            max_reasoning_steps=8
            )
        self.lr = lr
        self.carry = None
    
    def forward(self, batch):
        # Initialize carry if None
        # if self.carry is None:
        #     self.carry = self.model.initial_carry(batch)
        
        # HRM forward with carry state
        logits, num_steps = self.model(batch)
        return logits, num_steps
    
    def training_step(self, batch, batch_idx):
        # Reset carry periodically (every 10 steps)
        if batch_idx % 10 == 0:
            self.carry = None
            
        logits, num_steps = self(batch['inputs'])
        
        # Language modeling loss
        # logits = outputs['logits']  # [batch, seq, vocab]
        targets = batch['targets']   # [batch, seq]
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            targets.reshape(-1)
        )
        
        # ACT loss (if available)
        # if 'q_halt_logits' in outputs and 'target_q_continue' in outputs:
        #     q_loss = F.mse_loss(
        #         torch.sigmoid(outputs['q_continue_logits']), 
        #         outputs['target_q_continue']
        #     )
        #     loss = loss + 0.1 * q_loss
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits, num_steps = self(batch['inputs'])
        
        # Language modeling loss
        # logits = outputs['logits']  # [batch, seq, vocab]
        targets = batch['targets']   # [batch, seq]
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            targets.reshape(-1)
        )
        
        # Accuracy
        preds = logits.argmax(-1)
        acc = (preds == targets).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

# Training script
def train_hrm_sudoku():
    # Load dataset
    device = torch.device('cuda')
    
    ds = load_dataset("sapientinc/sudoku-extreme")
    
    # Create datasets
    train_dataset = SudokuDataset('train')
    val_dataset = SudokuDataset2('test')
    
    # CHANGE: Use same batch size as config
    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model with matching batch size
    model = HRMLightning(lr=1e-4)
    print("Initialized")
    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed',  # Faster training
        val_check_interval=0.25,  # Validate 4 times per epoch
        log_every_n_steps=1,
    )
    print("here")
    # Train
    trainer.fit(model, train_loader, val_loader)
    print("and here")

if __name__ == "__main__":
    train_hrm_sudoku() # HRM\epoch=0-step=59844.ckpt
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = HRMLightning.load_from_checkpoint("./HRM/epoch=0-step=59844.ckpt")
    # model.eval()
    # ds = load_dataset("sapientinc/sudoku-extreme")
    # tokenizer = {char: i for i, char in enumerate(".123456789")}
    # reverse_tokenizer = {i:char for i, char in enumerate(".123456789")}
    
    # print(ds['test']['question'][0])
    
    # tokenized = []
    # for c in ds['test']['question'][0]:
    #     tokenized.append(tokenizer[c])
    # print(tokenized)
    
    # tensor = torch.tensor([tokenized], dtype=torch.long).to(device)
    
    # out = model(tensor)
    # print(len(out))
    # print(out[1])
    
    # predicted_tokens = torch.argmax(out[0], dim=-1)
    # print(predicted_tokens[0])
    # print(ds['test']['answer'][0])
    
    # predicted_tokens = predicted_tokens[0].tolist()
    
    # actual_tokens = [tokenizer[char] for char in ds['test']['answer'][0]]
    
    
    # count = 0
    # for token, char in zip(predicted_tokens, actual_tokens):
    #     if token == char:
    #         count+=1
            
    # print("accuracy:", (count/len(predicted_tokens)))
    # print("count:", count)
    # print("total:", len(predicted_tokens))
    
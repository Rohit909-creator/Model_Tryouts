from datasets import load_dataset
import json
from datasets import load_dataset

tokenizer = {char:i for i, char in enumerate(".123456789")}

# Load Sudoku dataset
ds = load_dataset("sapientinc/sudoku-extreme")

# Make your tokenizer (char → id)
tokenizer = {char:i for i, char in enumerate(".123456789")}

# Function to tokenize one row
def tokenize_example(example):
    puzzle = example["question"]
    solution = example["answer"]

    # Convert chars to integer IDs
    puzzle_ids = [tokenizer[c] for c in puzzle]
    solution_ids = [tokenizer[c] for c in solution]

    return {
        "puzzle_ids": puzzle_ids,
        "solution_ids": solution_ids
    }

# Apply transformation
ds = ds.map(tokenize_example)

print(ds)
print(ds["train"][0])


import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import json
from datasets import load_dataset

# Your HRM model (paste your simplified model here)
from Model import HierarchicalReasoningModel_ACTV1

class SudokuDataset(Dataset):
    def __init__(self, data_file, ds, split='train'):
        self.tokenizer = {char: i for i, char in enumerate(".123456789")}
        
        # Load tokenized questions
        with open(data_file, 'r') as f:
            self.questions = [json.loads(line) for line in f]
        
        # Get answers from original dataset
        self.answers = []
        for i in range(len(ds[split])):
            tokenized_answer = [self.tokenizer[char] for char in ds[split][i]['answer']]
            self.answers.append(tokenized_answer)
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return {
            'inputs': torch.tensor(self.questions[idx], dtype=torch.long),
            'targets': torch.tensor(self.answers[idx], dtype=torch.long),
            'puzzle_identifiers': torch.tensor([0], dtype=torch.long)  # Single puzzle type for now
        }

class HRMLightning(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        
        # HRM config
        config = {
            'batch_size': 8,
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
        
        self.model = HierarchicalReasoningModel_ACTV1(config)
        self.lr = lr
        self.carry = None  # HRM state
    
    def forward(self, batch):
        # Initialize carry if None
        if self.carry is None:
            self.carry = self.model.initial_carry(batch)
        
        # HRM forward with carry state
        self.carry, outputs = self.model(self.carry, batch)
        return outputs
    
    def training_step(self, batch, batch_idx):
        # Reset carry periodically (every 10 steps)
        if batch_idx % 10 == 0:
            self.carry = None
            
        outputs = self(batch)
        
        # Language modeling loss
        logits = outputs['logits']  # [batch, seq, vocab]
        targets = batch['targets']   # [batch, seq]
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            targets.reshape(-1)
        )
        
        # ACT loss (if available)
        if 'q_halt_logits' in outputs and 'target_q_continue' in outputs:
            q_loss = F.mse_loss(
                torch.sigmoid(outputs['q_continue_logits']), 
                outputs['target_q_continue']
            )
            loss = loss + 0.1 * q_loss
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.carry = None  # Reset for clean validation
        outputs = self(batch)
        
        logits = outputs['logits']
        targets = batch['targets']
        
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
    ds = load_dataset("sapientinc/sudoku-extreme")
    
    # Create datasets
    train_dataset = SudokuDataset("Train_data.listl", ds, 'train')
    val_dataset = SudokuDataset("Train_data.listl", ds, 'test')  # You'll need val data too
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1)
    
    # Model
    model = HRMLightning(lr=1e-4)
    print("Initialized")
    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
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
    train_hrm_sudoku()
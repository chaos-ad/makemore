import notebooks.lecture1
import torch
from torch.nn import functional as F

def test_bigram():
    model = notebooks.lecture1.BigramModel(vocab_size=3)
    source = torch.tensor([0,1,2])
    target = torch.tensor([2,2,0])
    logodds = model(F.one_hot(source).float())

    loss = F.cross_entropy(logodds, target)
    loss.backward()
    print(logodds)
    print(loss)
    print(model.parameters())


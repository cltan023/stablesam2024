# StableSAM
Stablizing sharpness-aware training with a renormalization strategy

Usage:
```
from usam import USAM

# define a base optimizer as in standard training  
base_optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
optimizer = USAM(base_optimizer, rho=0.05, stable=True, adaptive=False)

for i in range(num_epochs):
    for data, target in train_dataloader:
        ...
        def closure():
            loss = train_loss_func(net(data), target)
            loss.backward()
            return loss
                       
        output = net(data)
        loss = train_loss_func(output, target)
        loss.backward()
        optimizer.step(closure)
        ...

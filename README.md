# dl-gpu-test
Test whether my environment can correctly use GPU for training and inference.
```
verify_pytorch.py
```
Intel Core i5 12490F cost 551 seconds

Nvidia RTX 3070 8G Desktop cost 19 seconds, 28x faster



```
python verify_pytorch.py

pytorch-gpu testing...
torch.cuda.is_available():True

Start time = 2024-03-06 22:10:15
We are using GPU now!!!
Epoch [5/1000], Loss: 0.59693241
Epoch [10/1000], Loss: 0.54300207
Epoch [15/1000], Loss: 0.53361923
Epoch [20/1000], Loss: 0.52981275
...
Epoch [990/1000], Loss: 0.51717466
Epoch [995/1000], Loss: 0.51717019
Epoch [1000/1000], Loss: 0.51716560
End time = 2024-03-06 22:10:35
Cost time = 19 seconds

Start time = 2024-03-06 22:10:35
We are using CPU now!!!
Epoch [5/1000], Loss: 0.61601233
Epoch [10/1000], Loss: 0.54237688
Epoch [15/1000], Loss: 0.52973700
Epoch [20/1000], Loss: 0.52588427
Epoch [25/1000], Loss: 0.52423447
Epoch [30/1000], Loss: 0.52330559
...
Epoch [955/1000], Loss: 0.51721621
Epoch [960/1000], Loss: 0.51721209
Epoch [965/1000], Loss: 0.51720792
Epoch [970/1000], Loss: 0.51720387
Epoch [975/1000], Loss: 0.51719981
Epoch [980/1000], Loss: 0.51719582
Epoch [985/1000], Loss: 0.51719189
Epoch [990/1000], Loss: 0.51718789
Epoch [995/1000], Loss: 0.51718408
Epoch [1000/1000], Loss: 0.51718020
End time = 2024-03-06 22:19:46
Cost time = 551 seconds
```
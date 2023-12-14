# MiniTorch Module 3

<img src="https://minitorch.github.io/_images/match.png" width="100px">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html

This module requires `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 2.

You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.
This assignment requires the following files from the previous assignments.

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py
        
simple layer = 2
![1637714238(1)](https://user-images.githubusercontent.com/89623727/143159465-7edf7e10-d6a3-45f6-8c87-cdc3f9cc2291.png)
![newplot (9)](https://user-images.githubusercontent.com/89623727/143159475-4d0a2719-f39d-4dd4-81f0-40689a7da6e2.png)
![1637714253(1)](https://user-images.githubusercontent.com/89623727/143159477-32c8a602-9bfd-4c4a-9ae8-a5eb15a756e6.png)

diag layer = 2
![1637714591(1)](https://user-images.githubusercontent.com/89623727/143159494-1cde00d7-2747-4504-854d-f9c116b2657a.png)
![newplot (10)](https://user-images.githubusercontent.com/89623727/143159501-f4a8a878-fd27-479a-8fde-a6c3b78a4a65.png)
![1637714597(1)](https://user-images.githubusercontent.com/89623727/143159511-34af39bc-7afa-4e2f-9a11-6d29c4c75e14.png)

split layer = 8
![1637714785(1)](https://user-images.githubusercontent.com/89623727/143159524-de9f2649-c21c-4e99-9d0d-b6c51e369060.png)
![1637714793(1)](https://user-images.githubusercontent.com/89623727/143159536-210e6528-840c-4cdb-88cb-4dceb7d0f451.png)

xor layer = 8
![1637715123(1)](https://user-images.githubusercontent.com/89623727/143159597-244461bb-98ad-4279-bd23-e1ec73092e0f.png)
![1637715132(1)](https://user-images.githubusercontent.com/89623727/143159605-b65c4c69-88a2-4708-a87a-d009ac36fcf2.png)

circle layer = 10
![1637715371(1)](https://user-images.githubusercontent.com/89623727/143159622-bb96b389-a181-44b6-b673-595f3038e8f7.png)
![1637715379(1)](https://user-images.githubusercontent.com/89623727/143159628-2889cafe-d54e-45ee-9fbf-0a851dc3b101.png)

spiral layer = 30 
![1637719507(1)](https://user-images.githubusercontent.com/89623727/143159799-412cab9d-f284-44fe-b013-113d9a37a518.png)
![1637719517(1)](https://user-images.githubusercontent.com/89623727/143159790-b80d8270-b467-4871-887c-7d82a1944b3f.png)

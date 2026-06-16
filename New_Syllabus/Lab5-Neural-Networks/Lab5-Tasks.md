# Lab 5: Neural Networks

## Task 1: 
```Python
# Define model
class LinearRegressionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(
                        in_feature=1, 
                        out_feature=1,
                        bias=True
                            )

    def forward(self, x):
        return self.linear(x)
```
- Use the above model definition and check if this model is capable of training all the datasets from [Lab 4](../Lab4-Regression-using-PyTorch). 
    - linear-regression-data1.csv
    - assignment-data.csv
    - assignment-data2.csv

Note: Just check if it is able to handle the data or not. 

## Task 2: 

1. Check if `LinearRegressionNN` model above is suitable for the dataset [binary_classification_moons.csv](../../Old_Syllabus/Lab3-NN/binary_classification_moons.csv). 
2. Check if the following models is suitable for the aforementioned dataset: [binary_classification_moons.csv](../../Old_Syllabus/Lab3-NN/binary_classification_moons.csv). Explain why and why not.  
```Python
class ModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5) 
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
    
    def forward(self, x):
        return self.layer_2(self.layer_1(x))

```
3. Add number of layers, or/and number of hidden units in the new model to check. An example is given in the [Lab-5.1-NN.ipynb file](./Lab-5.1-NN.ipynb). Explain. 

## Task 3: 
Task 3 will be mentioned during the lab session. 
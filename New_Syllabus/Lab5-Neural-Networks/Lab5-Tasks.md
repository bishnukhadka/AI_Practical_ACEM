# Lab 5: Neural Networks

## Task 1

```python
# Define model
class LinearRegressionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(
            in_features=1,
            out_features=1,
            bias=True
        )

    def forward(self, x):
        return self.linear(x)
```

* Using the model definition above, determine whether the model is capable of learning from each dataset provided in Lab 4:

  * `linear-regression-data1.csv`
  * `assignment-data.csv`
  * `assignment-data2.csv`

**Note:** Only determine whether the model can be trained on the datasets and whether its architecture is appropriate. A detailed performance evaluation is not required.

## Task 2

1. Determine whether the `LinearRegressionNN` model is suitable for the dataset `binary_classification_moons.csv`.

2. Determine whether the following model `ModelV0` is suitable for the same dataset. Justify your answer by explaining why the model is or is not appropriate.

```python
class ModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))
```

3. Modify the model by increasing the number of layers and/or hidden units. An example is provided in `Lab-5.1-NN.ipynb`. Evaluate the suitability of the resulting model architecture(s) for the `binary_classification_moons.csv` dataset and justify your conclusions.

## Task 3

Task 3 will be introduced during the laboratory session.

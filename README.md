# Simulated Data
>Simulated classification data with numpy

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

Controlling the generative process of data can allow for a deeper understanding of algorithm performance. This project is a modification of the simulated dataset code originally created by [dziganto](https://github.com/dziganto) in his [Synthetic_Dataset_Generation](https://github.com/dziganto/Synthetic_Dataset_Generation) repo. 

This project removes the pandas dataframe dependency in the original code. All generative operations are performed via the ndarray for speed and computational efficiency.

Performance when generating 5000 records:

* Old code =  20.6 seconds
* New code =  0.9 seconds

## Dataset Description
Feature # | Description | Important | Value
--- | --- | --- | ---
1|degree|Y|[(0=no bachelors, 8%), (1=bachelors, 70%), (2=masters, 80%), (3=PhD, 20%)]
2|age|N|[18, 60]
3|gender|N|[0=female, 1=male]
4|major|N|[0=anthropology, 1=biology, 2=business, 3=chemistry, 4=engineering, 5=journalism, 6=math, 7=political science]
5|GPA|N|[1.00, 4.00]
6|experience|Y|[(0-10, 90%), (10-25, 20%), (25-50, 5%)]\
7|bootcamp|Y|[(0=No, 25%), (1=Yes, 75%)]
8|GitHub|Y|[(0, 5%), (1-5, 65%), (6-20, 95%)]
9|blogger|Y|[(0=No, 30%), (1=Yes, 70%)]
10|blogs|N|[0, 20]

To determine if someone is hired, five target flags are generated stochastically based on the values of the dependent variables. The probabilistic flags are summed to dictate a hiring decision. To add prediction complexity, 5% of hiring decisions are flipped so they are incorrect.

A full description of this dataset is provided in [this](https://dziganto.github.io/data%20science/eda/machine%20learning/python/simulated%20data/Simulated-Datasets-for-Faster-ML-Understanding/) article.

### Prerequisites

Dependencies can be installed via:

```
pip install requirements.txt
```
## Author

* **Andrew Moss** - *Creator* - [agmoss](https://github.com/agmoss)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


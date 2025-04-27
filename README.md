# trackexp
trackexp is an experiment tracking module for Python for deep learning/machine learning usages.

```
trackexp.log("training", "loss", iter_index, loss_value)
                |          |         |                |
            context        |         |                |
         e.g.              |       "row id"           |
       "training"      name of     of tracked         |
     "validation"      tracked     data               |
   "testing"           data                     the data
                                                itself
```


Here is a birds-eye view of how it works.

``` python
import trackexp
trackexp.init()
experimental_config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
}
trackexp.metadata(experimental_config)
# [... inside your training loop ...]
trackexp.log("training", "loss", iter_index, loss_value)
# [... inside your validation loop ...]
trackexp.log("validation", "accuracy", iter_index, accuracy_value)
```

**Note**: the `trackexp.log(...)` and `trackexp.init(...)` do not even need to be in the same file, as long as `trackexp.init()` is reached first.



## Requirements

``` python
humanhash3
pandas
numpy
```

## Installation
Download directly from github:
``` bash
pip install git+https://github.com/YutongWangML/trackexp.git
```
Or if you wanna hack away, download then do
``` bash
git clone https://github.com/YutongWangML/trackexp.git
pip install --editable trackexp
```
The `--editable` flag lets you modify this package to your own custom needs.

## Saving Files
For data that should be saved to disk (like plots, model checkpoints, etc.), use the savefunc parameter:

``` python
def save_plot(context, name, identifier, data):
    filename = f"{context}_{name}_{identifier}.png"
    plt.figure()
    plt.plot(data)
    plt.savefig(filename)
    return filename

trackexp.log("analysis", "learning_curve", 1, learning_data, savefunc=save_plot)
```




# Why trackexp?

- Local: Everything is on on your computer. No web.

- Small: The code for the logging is all in `core.py` while the code for the post-analysis is all in `utils.py`.

- Hackable: If you use tensorboard, then one tricky thing is parsing the event file. With trackexp, it's as easy as

``` python
import trackexp
from trackexp.utils import get_data, get_metadata, list_experiments
import pandas as pd
import matplotlib.pyplot as plt

df_trn = get_data('iris_classification', context = 'training')
df_val = get_data('iris_classification', context = 'validation')
```

# License
MIT

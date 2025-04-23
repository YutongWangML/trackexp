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


## Requirements

``` python
humanhash3
pandas
numpy
```

## Installation

``` python
pip install --editable .
```

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
Local-first: Everything is stored locally by default
Lightweight: Small codebase that's easy to understand and modify
SQLite-based: Enables efficient querying and live tracking
General-purpose: Not tied to any specific ML framework

# License
MIT

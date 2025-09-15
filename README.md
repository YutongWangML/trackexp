# trackexp
trackexp is an experiment tracking module for Python for deep learning/machine learning usages.
Here's a very quick intro:

``` python
import trackexp as tx

# STEP 1: initialize trackexp:
tx.init(experiment_name = "training_recipe1")
#                                    ^
#                            where tracking data is stored
#                                [optional]
#                          if not supplied, then use humanhash3
#                          based on current timestamp

# STEP 2: crunch some numbers
# note: these are fake values for the sake of illustration
iter_index, loss_value = 0, 0.5 

# STEP 3: record the numbers
tx.log("training", "loss", iter_index, loss_value)
#           ^          ^         ^             ^
#        context \  name of  \ row ID     \  the data
#       e.g.      \  tracked  \ of tracked \   itself
#      "training"  \   data    \ data       \
#     "validation"  \           \ can be
#   "testing"                    \ "None"
#  "static_metrics"
# ... your choice

# STEP 4: a more realistic look
for t in range(5): # [fake training loop]
    tx.log("training", "loss", t, t**2)
    
    tx.start_timer('training', t) # [Option for tracking time. Creates a 'wallclocktime' tracked data under the hood. No need to "import time"]

    import time
    time.sleep( 0.1 ) # [simulate some intensive calculations]
    
    tx.stop_timer('training', t)

    # [remember to set your model to W for Wumbo, oops I mean model.eval()]
    tx.log("validation", "accuracy", t, 10*t)

# STEP 5: logging "static" metric
tx.log('static_metric', 'test_acc', None, 100)
df_train = tx.get_data('training_recipe1','training')
df_valid = tx.get_data('training_recipe1', 'validation')
df_test = tx.get_data('training_recipe1', 'static_metric')

# STEP 6: plot your results in the same jupyter notebook
import matplotlib.pyplot as plt
plt.plot(df_train['wallclocktime'], df_train['loss'])
plt.title(tx.get_current_experiment()['name'])
plt.show()
```


**Note**: the `tx.log(...)` and `tx.init(...)` do not even need to be in the same file. This is useful if you have a multi-file program. Just `import trackexp as tx` wherever you want to add track. Make sure during execution, `trackexp.init()` is reached first.



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

# Doc

## Init


``` python
def init(
    experiment_name: Optional[str] = None,
    base_dir: str = "trackexp_out",
    humanhash_words: int = 4,
    overwrite: bool = True,
    verbose: bool = False,
) -> str:
    """
    Initialize a new experiment with either a provided name or a human-readable hash name.

    Args:
        experiment_name: Optional custom name for the experiment. If provided, uses this name
                         instead of generating a hash-based name.
        base_dir: Base directory for all experiments.
        humanhash_words: Number of words to use for the human-readable hash name (if auto-generating).
        overwrite: If True and experiment_name is provided, deletes any existing directory with that name.
        verbose: If True, enables verbose logging of data points to console.

    Returns:
        Path to the experiment directory.
    """
```

## Log

``` python
def log(
    context: str,
    name: str,
    identifier: Hashable,
    data: Any
) -> None:
    """
    Log data for the current experiment.

    Args:
        context: The context (table) to log to.
        name: The name of the data point.
        identifier: A unique identifier for the row.
        data: The data to log.

    Example:
    trackexp.log("training", "loss", iter_index, loss_value)
                |          |         |                |
            context        |         |                |
         e.g.              |       "row id"           |
       "training"      name of     of tracked         |
     "validation"      tracked     data               |
   "testing"           data                     the data
                                                itself

    Note.1: you can store information inside
    trackexp.saved_vars = {}

    and use it like

    trackexp.saved_vars['iter'] = curr_iter

    Note.2: `None` is a perfectly acceptable identifier.
    You should use it for logging "constants" e.g., performance on test set at best validation loss.

    """
```



# Why trackexp?

- Local: Everything is on on your computer. No web.

- Small: The code for the logging is all in `core.py` while the code for the post-analysis is all in `utils.py`.

- Hackable: If you use tensorboard, then one tricky thing is parsing the event file. Parsing trackexp's data dump is simpler. For illustration, run `examples/neural_network_example.py`. Post-processing is as easy as

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

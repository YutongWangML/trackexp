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

## Quick usage

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

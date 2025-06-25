# sklearn-predict

* accepts: sdc.api.Spectrum2D
* generates: sdc.api.Spectrum2D

Loads a pickled scikit-learn model and generates predictions on the incoming data, adding it to the sample data.

```
usage: sklearn-predict [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                       [-N LOGGER_NAME] [--skip] -m MODEL_FILE -t TARGET

Loads a pickled scikit-learn model and generates predictions on the incoming
data, adding it to the sample data.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -m MODEL_FILE, --model_file MODEL_FILE
                        The pickled scikit-learn model to load and use.
                        (default: None)
  -t TARGET, --target TARGET
                        The sample data field to store the predictions under.
                        (default: None)
```

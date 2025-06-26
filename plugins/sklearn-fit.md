# sklearn-fit

* accepts: sdc.api.Spectrum2D

Builds a scikit-learn model on the incoming data and saves it.

```
usage: sklearn-fit [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                   [-N LOGGER_NAME] [--skip] [-m MODEL] [-p JSON] [-T FILE] -t
                   TARGET -o FILE

Builds a scikit-learn model on the incoming data and saves it.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -m MODEL, --model MODEL
                        The classname of the model to build. (default: None)
  -p JSON, --model_params JSON
                        The parameters of the model as JSON string. (default:
                        None)
  -T FILE, --template FILE
                        The path to the pickled model template to load and
                        train instead of classname/parameters. (default: None)
  -t TARGET, --target TARGET
                        The name of the sample data field to use as output
                        variable. (default: None)
  -o FILE, --output_file FILE
                        The file to save the model to. (default: None)
```

# A basic running example

This tutorial explore the process of how to use eprllib. Basically you need to follow four steps:

1. Configure your environment.
2. Register the environment.
3. Configure the algorithm.
4. Configure and run the training.

After that, you will have a trained policy that you can use in your application.

## 1. Configure your environment

To configure the environment first we construct an EnvConfig object. Ot contains all the information 
needed to comfigurate properly the environment.

```
from eprllib.Env.EnvConfig import EnvConfig

my_model = EnvConfig()
my_model.generals(
    epjson_path = '',
)
```
## 2. Register the environment



## 3. Configure the algorithm



## 4. Configure and run the training

Episodes API
============

Introduction
------------

In eprllib, **Episodes** define the configuration and setup of each simulation run. They provide a mechanism to customize the environment for each episode, allowing you to simulate different scenarios, load different building models or weather files, or use different schedules. This document provides a detailed explanation of the Episodes API in eprllib.

.. image:: Images/episodes.png
    :width: 600
    :alt: Episodes diagram
    :align: center
    :figclass: align-center
    :caption: Episodes diagram.

Defining Episodes with EnvironmentConfig
----------------------------------------

Episodes are defined within the ``EnvironmentConfig`` class using the ``episodes()`` method. This method allows you to specify:

*   ``episode_fn``: A function that defines the episode.
*   ``episode_fn_config``: A dictionary of parameters that will be passed to the episode function.

.. code-block:: python

    from eprllib.Environment.EnvironmentConfig import EnvironmentConfig

    env_config = EnvironmentConfig()
    env_config.episodes(
        episode_fn=lambda **kwargs: None,
        episode_fn_config={}
    )

Episode Functions (episode_fn)
------------------------------

Episode functions are responsible for setting up the environment at the beginning of each episode. They can be used to:

*   **Modify the EnergyPlus Model:** Change parameters in the EnergyPlus model (e.g., window properties, construction materials).
*   **Change the Weather File:** Load a different weather file for each episode.
*   **Load Different Schedules:** Use different occupancy, lighting, or equipment schedules.
*   **Define Start and End Conditions:** Set specific start and end times for the simulation.
*   **Perform Other Setup Tasks:** Any other tasks that need to be done at the beginning of an episode.

*   **Creating Custom Episode Functions:**

    You can create custom episode functions to implement any episode logic you need. An episode function should:

    *   Take the `kwargs` to receive the `episode_fn_config` dictionary.
    *   Return `None`.

Episode Function Configuration (episode_fn_config)
--------------------------------------------------

Episode functions can be configured using the ``episode_fn_config`` parameter in the ``episodes()`` method of ``EnvironmentConfig``. This allows you to customize the behavior of the episode function without modifying its code.

*   **Defining Custom Configuration Parameters:**

    When creating custom episode functions, you can define your own configuration parameters to control their behavior.

* **Configuring the episode function:**

    The episode function can be configured with any parameters.

Integration with the Environment
--------------------------------

Episode functions interact with the ``Environment`` class by modifying the EnergyPlus model or other environment settings before the episode begins.

*   **When Episode Functions Are Called:**

    Episode functions are called at the beginning of each episode, before the simulation starts.

Examples
--------

Here's a complete example of how to define and use an episode function:

.. code-block:: python

    from eprllib.Environment.EnvironmentConfig import EnvironmentConfig

    def my_episode_function(epjson_files_folder_path, epw_files_folder_path, load_profiles_folder_path, **kwargs):
        # Example: Load a different EnergyPlus model for each episode
        # ... load a different EnergyPlus model ...
        # Example: Load a different weather file for each episode
        # ... load a different weather file ...
        # Example: Load different schedules for each episode
        # ... load different schedules ...
        pass

    # Create the EnvironmentConfig object
    env_config = EnvironmentConfig()

    # Integrate the episode function into the environment configuration
    env_config.episodes(
        episode_fn=my_episode_function,
        episode_fn_config={
            'epjson_files_folder_path': "C:/Users/grhen/Documents/GitHub/SimpleCases/data/models",
            'epw_files_folder_path': "C:/Users/grhen/Documents/GitHub/SimpleCases/data/weathers",
            'load_profiles_folder_path': "C:/Users/grhen/Documents/GitHub/SimpleCases//data/schedules",
        }
    )

By understanding these concepts, you'll be able to effectively define and use episodes in eprllib for your building energy optimization and control projects.

Rewards API
===========

Introduction
------------

In ``eprllib``, ``Rewards`` are used to guide the learning process of Reinforcement Learning (RL) agents. They 
provide a numerical signal that indicates how well an agent is performing in the environment. The ``Rewards`` API 
provides a structured way to define and configure these reward functions. This document provides a detailed 
explanation of the ``Rewards`` API in ``eprllib``.

.. image:: Images/rewards.png
    :width: 600
    :alt: Rewards diagram
    :align: center


Reward functions are responsible for calculating the reward that the agent receives at each time step. They take into 
account the current state of the environment, the agent's observations, and the agent's actions to determine the reward.

*   **Common Reward Function Examples:**

    *   **Energy Consumption:** A reward function that penalizes high energy consumption.
    *   **Thermal Comfort:** A reward function that rewards the agent for maintaining comfortable temperatures.
    *   **Combined Energy and Comfort:** A reward function that balances energy consumption and thermal comfort.
    *   **Custom Metrics:** Reward functions can be created to optimize any metric available in the EnergyPlus simulation.

*   **Creating Custom Reward Functions:**

    You can create custom reward functions to implement more complex reward logic. A reward function should:

    *   Take the agent's name, the thermal zone, and any other relevant parameters as input.
    *   Take the `kwargs` to receive the `info` dictionary.
    *   Return a numerical value representing the reward.


Relationship with Observations, Actions, and Environment State
--------------------------------------------------------------

Reward functions use information from the environment, the agent's observations, and the agent's actions to calculate the reward. The flow of information is as follows:

1.  **Environment State:** The reward function has access to the current state of the EnergyPlus environment.
2.  **Agent's Observations:** The reward function can use the agent's observations (as defined in ``ObservationSpec``).
3.  **Agent's Actions:** The reward function can take into account the actions that the agent has taken.
4.  **Reward Calculation:** The reward function combines this information to calculate a numerical reward value.

Using Infos in the Reward Function
----------------------------------

The `info` dictionary, returned by the `step()` method of the `Environment`, can be used to access additional information within the reward function. This allows you to create more complex and informative reward signals.

*   **Accessing the `info` Dictionary:**

    The `info` dictionary is passed to the reward function as part of the `kwargs`. You can access it like this:

    .. code-block:: python

        def my_reward_function(agent_name, thermal_zone, beta, people_name, cooling_name, heating_name, cooling_energy_ref, heating_energy_ref, **kwargs):
            info = kwargs.get("info", {})
            # ... use info to calculate the reward ...

*   **Example of Using `info`:**

    You might use `info` to access:

    *   Energy consumption data.
    *   Thermal comfort metrics.
    *   Weather data.
    *   Any other information that is available in the `info` dictionary.

Custom ``Reward`` functions
----------------------------

Examples
--------

Here's a complete example of how to define and use a reward function:

.. code-block:: python

    
Connectors API
==============

Introduction
------------

In eprllib, ``Connectors`` define how agents interact with the environment and with 
each other. They provide a flexible mechanism to implement various interaction 
patterns, such as cooperation, competition, or hierarchical control. This document 
provides a detailed explanation of the ``Connectors`` API in ``eprllib``.

.. .. figure:: Images/connectors.png
..     :width: 600
..     :alt: Connectors diagram
..     :align: center

..    *Figure 1: Schematic representation of the connectors function.*



Creating custom Connector functions
------------------------------------------

The ``Connector`` functions are responsible for defining the final observations of agents,
considering not only the observation obtained from a particular agent but all the presents in the environment. 
To define a custom ``Connector`` function, you need to follow these steps:

1. Override the ``setup(self)`` method.
2. Override the ``get_agent_obs_dim(self, agent: str)`` method.
3. Override the ``get_agent_obs_indexed(self, agent: str)`` method.
4. Optionally, you can override the methods ``set_top_level_obs(...)`` and ``set_low_level_obs(...)``.

.. note:: Use the decorator ``override`` in each method.``

.. warning:: By default, all the agents are considered Top Level agents using the filtered observation space
    directly. For more complex interaction patterns, such as hierarchical control or exchange of information, 
    it is necessary to override the ``set_top_level_obs(...)`` and ``set_low_level_obs(...)`` methods.

First, it is necessary to create a new class that inherits from ``BaseConnector``. Then, you can 
override the necessary methods to define the behavior of your custom connector. Here is an 
example of how to create a custom connector:

.. code:: python

    from eprllib.Connectors.BaseConnector import BaseConnector

    class MyCustomConnector(BaseConnector):
        ...

The ``setup`` method is used to perform any necessary setup tasks for the connector. Here is 
an example of how to override the setup method:

.. code:: python

    from typing import Dict, Any
    from eprllib.Utils.annotations import override

    @override(BaseConnector)
    def setup(self) -> None:

        # We will use this attribute to store a flag to init the observation index.
        self.obs_index_init: Dict[str, bool] = {
            "Ventilation Agent": False
        }


Now, we need to override the ``get_agent_obs_dim(self, agent: str)`` method, which is called when the 
environment is initialized and after the ``Connector`` functions are initialized. Here, we need to 
define the observation space for each agent. The observation space should be defined as a Open AI Gymnasium space.
It is possible to define a different observation space for each agent, but in this example, we will 
define the same observation space for all agents.

.. code:: python
    
    @override(BaseConnector)
    def get_agent_obs_dim(
        self,
        env_config: Dict[str,Any],
        agent: str
        ) -> spaces.Box:
        """
        Get the agent observation dimension.
        
        Args:
            agent (str): Agent identifier.
        
        Returns:
            gym.spaces.Space: Agent observation dimension.
            
        Raises:
            NotImplementedError: If the method is not implemented in the child class.
        """
        # Mark the flag as True.
        self.obs_index_init[agent] = True
        
        # This is the orther that the Filter will use to create the observation array. It 
        # is important to keep this order when creating the observation array to allow a 
        # good interpretability of the observation space.
        self.obs_indexed[agent] = {            
            f"{agent}: Indoor Setpoint-Mean Indoor Air": 0,
            f"{agent}: Outdoor Air Temperature": 1,
            f"{agent}: Wind Speed": 2,
            f"{agent}: Wind Direction": 3,
            f"{agent}: Occupancy status": 4,
            f"{agent}: Energy for heating": 5,
            f"{agent}: Energy for cooling": 6,
            f"{agent}: North Actuator Status": 7,
            f"{agent}: South Actuator Status": 8,
        }

        # We use the dict len to define the observation space dimension for this agent.
        obs_space_len: int = len(self.obs_indexed[agent])
        
        # Build and return the observation space for this agent.
        return spaces.Box(
                low = -np.inf, 
                high = np.inf, 
                shape=(obs_space_len, ),
                dtype=np.float32
            )
        
In this example, we use the ``get_agent_obs_dim`` method to create the obs indexes for each agent. To do that only once, 
we use a flag to check if the obs index has been initialized for each agent. In concordance with this, we override the 
``get_agent_obs_indexed(self, agent: str)`` method to return the obs index for each agent. Here is an example of how 
to override this method:

.. code:: python

    @override(BaseConnector)
    def get_agent_obs_indexed(
        self,
        env_config: Dict[str, Any],
        agent: str
        ) -> Dict[str, int]:
        """
        Get a dictionary of the agent observation parameters and their respective index in the observation array.
        
        Args:
            agent (str): Agent identifier.
            
        Returns:
            Dict[str, int]: Dictionary of the agent observation parameters and their respective index in the observation array.
        
        Raises:
            NotImplementedError: If the method is not implemented in the child class.

        """
        if not self.obs_index_init[agent]:
            self.get_agent_obs_dim(env_config, agent)
        return self.obs_indexed[agent]
    

Advance usage of Connector functions
------------------------------------------

When heriachical control is implemented in the environment, it is possible to use the 
``set_top_level_obs(...)`` and ``set_low_level_obs(...)`` methods to define the multiagent 
observation. Both methods togheter with the ``get_agent_obs_dim(...)`` and ``get_agent_obs_indexed(...)`` methods 
allow to define a hierarchical control structure in the environment.

.. warning:: THIS SECTION IS STILL IN DEVELOPMENT AND MAY CHANGE IN THE FUTURE.


.. code:: python

    def set_top_level_obs(
        self,
        agent_states: Dict[str, Dict[str, Any]],
        dict_agents_obs: Dict[str, Any],
        infos: Dict[str, Dict[str, Any]],
        is_last_timestep: bool = False
        ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]:
        """
        Set the multi-agent observation.
        
        Args:
            agent_states (Dict[str, Dict[str, Any]]): Agent states.
            dict_agents_obs (Dict[str, Any]): Dictionary of agents' observations.
            infos (Dict[str, Dict[str, Any]]): Additional information.
            is_last_timestep (bool, optional): Flag indicating if it is the last timestep. Defaults to False.
        
        Returns:
            Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]: Multi-agent observation, updated infos, and a flag indicating if it is the lowest level.
        
        """
        is_lowest_level = True
        return dict_agents_obs, infos, is_lowest_level
    
    
    def set_low_level_obs(
        self,
        agent_states: Dict[str,Dict[str,Any]],
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]],
        goals: Dict[str, Any]
        ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]:
        """
        Set the multiagent observation.
        
        Args:
            agent_states (Dict[str, Dict[str, Any]]): Agent states.
            dict_agents_obs (Dict[str, Any]): Dictionary of agents' observations.
            infos (Dict[str, Dict[str, Any]]): Additional information.
            goals (Dict[str, Any]): Goals.
        
        Returns:
            Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]: Multi-agent observation, updated infos, and a flag indicating if it is the lowest level.
          
        """
        is_lowest_level = True
        return dict_agents_obs, infos, is_lowest_level

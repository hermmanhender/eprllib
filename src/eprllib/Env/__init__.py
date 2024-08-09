"""
Environments
============

The environments in RLlib can be implemented as a simple-agent, multi-agent or
external environment. In eprllib we use a multi-agent aproach, allowing to run 
multiples agents in the environment, but also a single agent if only one is specify.

The standard configuration use a policy with fully shared parameters, but in the next
versions we hope to add flexibility to the policy.

You can configure the environment with the :class:`~eprllib.Env.EnvConfig.EnvConfig` class .
"""

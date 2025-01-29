"""
Annotations
============


"""

def override(parent_cls):
    """Decorator for documenting method overrides.

    Args:
        parent_cls: The superclass that provides the overridden method. If
            `parent_class` does not actually have the method or the class, in which
            method is defined is not a subclass of `parent_class`, an error is raised.

    .. testcode::
        :skipif: True

        from ray.rllib.policy import Policy
        class TorchPolicy(Policy):
            ...
            # Indicates that `TorchPolicy.loss()` overrides the parent
            # Policy class' own `loss method. Leads to an error if Policy
            # does not have a `loss` method.

            @override(Policy)
            def loss(self, model, action_dist, train_batch):
                ...

    """

    class OverrideCheck:
        def __init__(self, func, expected_parent_cls):
            self.func = func
            self.expected_parent_cls = expected_parent_cls

        def __set_name__(self, owner, name):
            # Check if the owner (the class) is a subclass of the expected base class
            if not issubclass(owner, self.expected_parent_cls):
                raise TypeError(
                    f"When using the @override decorator, {owner.__name__} must be a "
                    f"subclass of {parent_cls.__name__}!"
                )
            # Set the function as a regular method on the class.
            setattr(owner, name, self.func)

    def decorator(method):
        # Check, whether `method` is actually defined by the parent class.
        if method.__name__ not in dir(parent_cls):
            raise NameError(
                f"When using the @override decorator, {method.__name__} must override "
                f"the respective method (with the same name) of {parent_cls.__name__}!"
            )

        # Check if the class is a subclass of the expected base class
        OverrideCheck(method, parent_cls)
        return method

    return decorator
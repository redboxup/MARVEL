from typing import Dict, Any

CONFIG_REGISTRY = {}


class BaseConf:
    registry: Dict[str, Any] = CONFIG_REGISTRY

    def __init_subclass__(cls, group=None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            key = cls.name  # type: ignore
            group_key = group or cls.__module__.split(".")[-1]
            cls.registry.setdefault(group_key, {})[key] = cls

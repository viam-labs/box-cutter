import asyncio

from viam.module.module import Module

# Imported for their side effect: defining each model registers it with the
# registry that run_from_registry serves.
from models.control import Control as ControlModel  # noqa: F401

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())

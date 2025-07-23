from contextlib import AbstractContextManager
from types import TracebackType
from typing import Protocol

from .action import ActionLabel


class ActionInvoker(Protocol):
    def waiting_for(self, label: ActionLabel) -> AbstractContextManager: ...

    def runtarget_args(self) -> list[str]: ...


class DummyInvoker(ActionInvoker):
    def waiting_for(self, label: ActionLabel) -> AbstractContextManager:
        class WaitingFor:
            def __enter__(this):
                return this

            def __exit__(
                this,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> None:
                pass

        return WaitingFor()

    def runtarget_args(self):
        return []

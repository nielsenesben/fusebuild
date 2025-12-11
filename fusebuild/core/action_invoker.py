from contextlib import AbstractContextManager
from types import TracebackType
from typing import Protocol

from .action import ActionLabel


class WaitingFor(Protocol):
    def __enter__(this) -> "WaitingFor": ...

    def __exit__(
        this,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...


class ActionInvoker(Protocol):
    def waiting_for(self, label: ActionLabel) -> AbstractContextManager[WaitingFor]: ...

    def runtarget_args(self) -> list[str]: ...


class DummyInvoker(ActionInvoker):
    def waiting_for(self, label: ActionLabel) -> AbstractContextManager[WaitingFor]:
        class WaitingForImpl(WaitingFor):
            def __enter__(this) -> "WaitingFor":
                return this

            def __exit__(
                this,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> None:
                pass

        return WaitingForImpl()

    def runtarget_args(self) -> list[str]:
        return []

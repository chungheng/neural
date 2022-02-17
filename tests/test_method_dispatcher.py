import pytest
from neural._method_dispatcher import MethodDispatcher


class BackendClass1:
    def func(self, arg):
        return 1  # base implementation


class BackendClass2:
    def func(self, arg):
        return 2  # base implementation


class BackendClass3:
    def func(self, arg):
        return 3  # base implementation


class Klass:
    backend = None

    def __init_subclass__(cls) -> None:
        if not hasattr(cls.func, "register"):
            func = MethodDispatcher(cls.func)
            MethodDispatcher.__set_name__(func, cls, "func")
            cls.func = func

    @MethodDispatcher
    def func(self, arg):
        return 0  # base implementation

    def set_backend(self, BackendClass):
        self.backend = BackendClass()
        if callable(method := getattr(BackendClass, "func", None)):
            self.func.register(BackendClass, method)


class SubKlass(Klass):
    backend = None

    def func(self, arg):
        return 4  # base implementation


def test_MethodDispatch():
    c = Klass()
    assert hasattr(c.func, "register")
    assert c.func(0) == 0
    c.set_backend(BackendClass1)
    assert hasattr(c.func, "register")
    assert c.func(0) == 1
    c.set_backend(BackendClass2)
    assert hasattr(c.func, "register")
    assert c.func(0) == 2
    c.set_backend(BackendClass3)
    assert hasattr(c.func, "register")
    assert c.func(0) == 3

    # FIXME: Is there a way to avoid writing to class's attribute at all?
    # Note that while this Klass has registry
    # the Klass below does not have a registry. It is not shared between them
    assert len(Klass.func.registry) == 3

    d = Klass()
    assert len(Klass.func.registry) == 3
    assert d.func(0) == 0

    e = SubKlass()
    assert not SubKlass.func.registry
    assert hasattr(e.func, "dispatch")
    assert hasattr(e.func, "register")
    assert e.func(0) == 4

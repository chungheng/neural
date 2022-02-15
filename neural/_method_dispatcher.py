from types import MappingProxyType, MethodType
import weakref
from abc import get_cache_token
from functools import update_wrapper

class MethodDispatcher:
    """Dispatch Neural.Model method descriptor.

    Example:

        The MethodDispatcher class can be used as a decorator
        for methods of the class as follows::

            class Klass:
                backend = None
                def __init_subclass__(cls) -> None:
                    if not hasattr(cls.func, 'dispatch'):
                        setattr(cls, 'func', MethodDispatcher(cls.func))

                @MethodDispatcher
                def func(self, arg):
                    return 0 # base implementation

                def set_backend(self, BackendClass):
                    self.backend = BackendClass()
                    if callable(method := getattr(BackendClass, 'func', None)):
                        self.func.register(BackendClass, method)

    """

    def __init__(self, func):
        if not callable(func) and not hasattr(func, "__get__"):
            raise TypeError(f"{func!r} is not callable or a descriptor")

        self.cache_token = None
        self.registry =  None
        self.dispatch_cache = None
        func.register = self.register
        func.registry = self.registry
        func.dispatch = self.dispatch
        self.func = func

    def __set_name__(self, owner, name):
        self.registry = {}
        self.dispatch_cache = weakref.WeakKeyDictionary()
        self.dispatch_cache[object] = self.func

        self.func.register = self.register
        self.func.registry = MappingProxyType(self.registry)
        self.func.dispatch = self.dispatch
        self.func._clear_cache = self.dispatch_cache.clear

    def dispatch(self, BackendCls):
        if self.cache_token is not None:
            current_token = get_cache_token()
            if self.cache_token != current_token:
                self.dispatch_cache.clear()
                self.cache_token = current_token
        try:
            impl = self.dispatch_cache[BackendCls]
        except KeyError:
            try:
                impl = self.registry[BackendCls]
            except KeyError:
                impl = self.func
            self.dispatch_cache[BackendCls] = impl
        return impl

    def register(self, BackendCls, method):
        """generic_method.register(BackendCls, func) -> func

        Registers a new implementation for the given *BackendCls* on a *generic_method*.
        """
        if BackendCls is None:
            BackendCls = object
        method.__isabstractmethod__ = self.__isabstractmethod__
        method.register = self.register
        update_wrapper(method, self.func)
        self.registry[BackendCls] = method
        if self.cache_token is None:
            self.cache_token = get_cache_token()
        self.dispatch_cache.clear()
        return method

    def __get__(self, obj, cls=None):
        # if called from class, return default implementation
        if obj is None:
            return self.func
        # if called from instance return dispatched implementation
        return MethodType(self.dispatch(obj.backend.__class__), obj)

    @property
    def __isabstractmethod__(self):
        return getattr(self.func, '__isabstractmethod__', False)
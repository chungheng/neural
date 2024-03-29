try:
    from types import SimpleNamespace
except:

    class SimpleNamespace(object):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __repr__(self):
            keys = sorted(self.__dict__)
            items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
            return "namespace({})".format(", ".join(items))

        def __eq__(self, other):
            return self.__dict__ == other.__dict__

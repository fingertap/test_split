class Registry:
    def __init__(self):
        self.class_mapping = {}

    def register(self, field='default', name=None):
        def _register(cls):
            if field not in self.class_mapping:
                self.class_mapping[field] = {}
            module_name = name or cls.__name__
            self.class_mapping[field][module_name] = cls
            return cls
        return _register

    def build(self, name=None, params: dict=None, field='default'):
        return self.class_mapping[field][name](**params)


MODULES = Registry()
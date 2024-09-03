import pkgutil

__all__ = []
for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
    if is_pkg or "." in module_name:
        continue
    __all__.append(module_name)
    _module = loader.find_spec(module_name).loader.load_module()
    globals()[module_name] = _module
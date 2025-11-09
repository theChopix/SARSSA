import importlib

class PluginManager:
    @staticmethod
    def load(plugin_name):
        module_path = f"plugins.{plugin_name}"
        module = importlib.import_module(module_path)
        return module.Plugin()
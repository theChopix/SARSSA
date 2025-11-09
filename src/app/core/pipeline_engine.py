from app.core.plugin_manager import PluginManager

class PipelineEngine:
    def __init__(self, steps):
        self.steps = steps

    def run(self, context):
        for step in self.steps:
            plugin_name = step["plugin"]
            params = step.get("params", {})
            plugin = PluginManager.load(plugin_name)
            context = plugin.run(context, **params)
        return context
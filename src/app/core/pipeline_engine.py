from app.core.plugin_manager import PluginManager

class PipelineEngine:
    def __init__(self, steps):
        self.steps = steps

    def run(self, context):
        for step in self.steps:
            plugin = PluginManager.load(step["plugin"])
            context = plugin.run(context)
        return context
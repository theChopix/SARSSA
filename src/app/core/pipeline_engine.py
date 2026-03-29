import mlflow
import datetime

from app.core.plugin_manager import PluginManager

class PipelineEngine:
    def __init__(self, steps):
        self.steps = steps

    def run(self, context):

        mlflow.set_experiment("pipeline_experiments_01")

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        with mlflow.start_run(run_name=f"pipeline_run_{timestamp}") as _:

            for step in self.steps:
                plugin_name = step["plugin"]
                params = step.get("params", {})
                plugin = PluginManager.load(plugin_name)

                with mlflow.start_run(run_name=f"{plugin_name}", nested=True) as pipeline_step:

                    plugin.run(context, **params)

                    plugin_category = plugin_name.split(".")[0]
                    context[plugin_category] = {'run_id': pipeline_step.info.run_id}

        return context
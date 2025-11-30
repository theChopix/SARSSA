import mlflow
import datetime

from app.core.plugin_manager import PluginManager

class PipelineEngine:
    def __init__(self, steps):
        self.steps = steps

    def run(self, context):

        mlflow.set_experiment("pipeline_experiments_01")

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        with mlflow.start_run(run_name=f"pipeline_run_{timestamp}") as pipeline_experiment:

            for step in self.steps:
                plugin_name = step["plugin"]
                params = step.get("params", {})
                plugin = PluginManager.load(plugin_name)

                with mlflow.start_run(run_name=f"{plugin_name}", nested=True) as pipeline_step:

                    context = plugin.run(context, **params)
                    context['last_plugin_run_id'] = pipeline_step.info.run_id

        return context
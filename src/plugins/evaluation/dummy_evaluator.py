from plugins.plugin_interface import BasePlugin
from app.utils.logger import logger

class Plugin(BasePlugin):
    def run(self, context):
        logger.info("Running dummy evaluation...")
        context["evaluation"] = {"RMSE": 0.12}
        return context

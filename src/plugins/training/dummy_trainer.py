from plugins.plugin_interface import BasePlugin
from app.utils.logger import logger

class Plugin(BasePlugin):
    def run(self, context):
        logger.info("Running dummy training...")
        context["model"] = {"status": "trained", "accuracy": 0.85}
        return context
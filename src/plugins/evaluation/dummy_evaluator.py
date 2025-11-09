from app.utils.logger import logger

class Plugin:
    def run(self, context):
        logger.info("Running dummy evaluation...")
        context["evaluation"] = {"RMSE": 0.12}
        return context

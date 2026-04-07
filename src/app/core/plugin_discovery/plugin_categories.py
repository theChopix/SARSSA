"""Static registry of plugin categories and their metadata."""

from app.models.plugin import CategoryInfo, CategoryType

PLUGIN_CATEGORIES: dict[str, CategoryInfo] = {
    "dataset_loading": CategoryInfo(
        order=0, type=CategoryType.ONE_TIME, display_name="Dataset Loading"
    ),
    "training_cfm": CategoryInfo(order=1, type=CategoryType.ONE_TIME, display_name="Training CFM"),
    "training_sae": CategoryInfo(order=2, type=CategoryType.ONE_TIME, display_name="Training SAE"),
    "neuron_labeling": CategoryInfo(
        order=3, type=CategoryType.ONE_TIME, display_name="Neuron Labeling"
    ),
    "labeling_evaluation": CategoryInfo(
        order=4, type=CategoryType.MULTI_RUN, display_name="Labeling Evaluation"
    ),
    "inspection": CategoryInfo(order=5, type=CategoryType.MULTI_RUN, display_name="Inspection"),
    "steering": CategoryInfo(order=6, type=CategoryType.MULTI_RUN, display_name="Steering"),
}

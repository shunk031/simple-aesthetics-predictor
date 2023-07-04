import logging

logger = logging.getLogger(__name__)


def get_model_name(org_model_name: str, version: int) -> str:
    org, model_name = org_model_name.split("/")
    logger.debug(f"org: {org}, model name: {model_name}")

    model_name = "-".join(model_name.split("-")[1:])
    return f"shunk031/aesthetics-predictor-v{version}-{model_name}"

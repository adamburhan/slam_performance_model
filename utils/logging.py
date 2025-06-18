import logging
import wandb

def get_logger(config):
    """
    Set up logging based on configuration.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        logger: Configured logger object.
    """
    # Create a logger
    logger = logging.getLogger(config['experiment_name'])
    logger.setLevel(logging.INFO)

    # Create a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create a formatter and set it for the console handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    # If W&B is enabled, initialize it
    if config['logging']['use_wandb']:
        wandb.init(project=config['logging']['wandb_project'],
                   entity=config['logging']['wandb_entity'],
                   config=config,
                   name=config['experiment_name'])
        logger.info("Weights & Biases logging enabled")

    return logger

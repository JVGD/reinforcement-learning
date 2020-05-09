import logging


def get_logger(logger_name=None, level=logging.INFO):
    """Getting nice logger

    Parameters
    ----------
    logger_name : str
        Logger name, should be __name__ or None
    level : int, optional
        Logging level, by default logging.INFO

    Returns
    -------
    logging.Logger
        Logger
    """
    # Getting logger
    logger = logging.getLogger(logger_name)
    
    # Basic logging conf
    logging_format = '%(asctime)s %(module)s [%(levelname)s]: %(message)s'
    logging.basicConfig(level=level, format=logging_format)
    return logger
from logger import logger, get_logger
logger.info("import ok: top-level logger exists")
get_logger("main").info("child logger ok")
print("python import test passed")

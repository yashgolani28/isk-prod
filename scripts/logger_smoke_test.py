from logger import setup_logger
log = setup_logger("smoke", r"system-logs\smoke.log")
log.info("smoke test: rotating file handler with 3-day retention is active")
print("ok")

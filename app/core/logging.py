import logging
import sys
from pythonjsonlogger import jsonlogger
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True

def configure_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    logHandler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s')
    logHandler.setFormatter(formatter)
    logHandler.addFilter(RequestIdFilter())
    logger.addHandler(logHandler)

logger = logging.getLogger("support_docs_copilot")

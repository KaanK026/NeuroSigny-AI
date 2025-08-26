import logging
from logging.handlers import QueueHandler
import multiprocessing
import sys
import traceback



# Logger

def logger_process(log_queue: multiprocessing.Queue, log_file: str = "camera_log.txt"):

    # File handler
    handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)


    logger = logging.getLogger("asl_logger")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    while True:
        try:
            record = log_queue.get()
            if record is None:  # Sentinel
                break
            logger.handle(record)
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception:
            traceback.print_exc(file=sys.stderr)


# Creating a queue logger for child processes
def get_queue_logger(name: str, log_queue: multiprocessing.Queue):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    qh = QueueHandler(log_queue)
    logger.addHandler(qh)
    return logger


if __name__ == "__main__":
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=logger_process, args=(q,))
    p.start()

    logger = get_queue_logger("test_logger", q)
    logger.info("Test log entry")

    q.put(None)
    p.join()

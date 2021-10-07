import time
import sys


STREAM = None


def format_msg(msg):
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    return f"{time_string} - INFO - {msg}"

def basicConfig(stream):
    global STREAM
    STREAM = stream

def info(msg):
    global STREAM
    STREAM.write(format_msg(msg) + "\n")
    STREAM.flush()
    sys.stdout.write(format_msg(msg) + "\n")
    sys.stdout.flush()
import time


class FlushingPrintingFile(object):
    def __init__(self, path, mode, old_stdout):
        self.path = path
        self.mode = mode
        self.old_stdout = old_stdout
        self.f = open(path, mode)
        self.last_written = time.time()

    def write(self, data):
        self.f.write(data)
        if time.time() - self.last_written > 15: # For google drive to syncronize
            self.f.flush()
            self.last_written = time.time()

        self.old_stdout.write(data)

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()

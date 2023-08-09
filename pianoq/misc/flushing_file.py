import traceback
import time


class FlushingPrintingFile(object):
    def __init__(self, path, mode, old_stdout):
        self.path = path
        self.mode = mode
        self.old_stdout = old_stdout
        self.f = open(path, mode)
        self.last_written = time.time()

    def write(self, data):
        self.old_stdout.write(data)
        self.f.write(data)
        if time.time() - self.last_written > 15:  # For google drive to syncronize
            self.f.flush()
            self.last_written = time.time()

    def flush(self):
        try:
            self.f.flush()
        except Exception as e:
            print('Exception!!!')
            print(e)
            traceback.print_exc()

    def close(self):
        try:
            self.flush()
        except Exception as e:
            print('Exception!!!')
            print(e)
            traceback.print_exc()
        try:
            self.f.close()
        except Exception as e:
            print('Exception!!!')
            print(e)
            traceback.print_exc()


from functools import wraps

def retry_if_exception(ex=Exception, max_retries=3):
    def outer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            assert max_retries > 0
            x = max_retries
            while x:
                try:
                    return func(*args, **kwargs)
                except ex:
                    print(f'Failed. {x} tries remain')
                    x -= 1
                    if x < 1:
                        raise
        return wrapper
    return outer

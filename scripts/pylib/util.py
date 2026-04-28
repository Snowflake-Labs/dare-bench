from functools import wraps


def strip_whitespace(func):
    """
    Decorator to strip whitespace from the return value of a function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Execute the original function and get its return value
        result = func(*args, **kwargs)
        # Check if the result is a string and strip whitespace if so
        if isinstance(result, str):
            return result.strip()
        return result

    return wrapper


def string_from_file(filepath: str) -> str:
    """
    open the file in read mode, read the string, and close the file handler
    """
    with open(filepath, "r") as f:
        return f.read()


def remove_http_scheme(url: str):
    if "https://" in url:
        return url.replace("https://", "")
    elif "http://" in url:
        return url.replace("http://", "")
    else:
        return url

import sys
import traceback

def error_message_detail(error, error_detail: sys):
    """
    Extracts and formats error details from the exception traceback.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error occurred in script: [{file_name}] at line: [{line_number}] error: [{error}]"

class CustomException(Exception):
    """
    Custom Exception class that formats error messages with detailed traceback info.
    """
    def __init__(self, error, error_detail: sys):
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message
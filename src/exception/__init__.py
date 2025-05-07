import sys
import logging


def error_message(error:Exception,error_detail:sys):
    _,_,traceback = error_detail.exc_info()

    file_name = traceback.tb_frame.f_code.co_filename
    line_number = traceback.tb_lineno

    message = f"The error occured in [{file_name}] at line {line_number}] : {str(error)}"

    logging.error(message)

    return message


class CustomException(Exception):
    def __init__(self,error:str,error_detail:sys):
        '''
        error        : the exception occured 
        error_detail :  details of the error i.e., file,line,type of error etc.
        '''
        # initialize the attributes of the parent(Exception) class
        super().__init__(error)

        self.message = error_message(error=error,error_detail=error_detail)

    def __str__(self):
        '''
        Return the string representation of the error message 
        '''

        return self.message
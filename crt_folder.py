#import modulue os
import os
 
def crt_folder(path):
    #check whether the directory exists or not
    #exist：True
    #not：False
    folder = os.path.exists(path)
 
    #result of check
    if not folder:
        #if not exist
        os.makedirs(path)
# import logging
#
# logging.basicConfig(filename='train_info.log', level=logging.DEBUG)
# logger = logging.getLogger().addHandler(logging.NullHandler())
# logging.setLevel(logging.DEBUG)
# # logger.setLevel(logging.DEBUG)
# logger.warning('Teste')

import logging
logging.basicConfig(filename='train_info.log', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.warning('Watch out!')  # will print a message to the console
logging.info('I told you so')  # will not print anything

# def main():
#     logging.basicConfig(filename='myapp.log', level=logging.INFO)
#     logger = logging.getLogger().addHandler(logging.NullHandler())
#     logger.info('Started')
#     # mylib.do_something()
#     logging.info('Finished')
#
#
# if __name__ == '__main__':
#     main()

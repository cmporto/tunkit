



class GeneralLogger:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.messages = []

    # careful cause the separator acts as a print (with end='\n')
    def log(self, *args, separator='\n'):
        """Prints if verbose and saves the message to the internal list."""
        msg = separator.join(str(arg) for arg in args)
        if self.verbose:
            print(msg)
        self.messages.append(msg)

    def get_log(self):
        """Returns the logged messages in full as a single string."""
        return '\n'.join(self.messages)
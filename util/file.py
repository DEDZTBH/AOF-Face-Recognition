import pickle
from os import path


def save(stuff, filename, ext='pkl', folder='data'):
    with open(path.join(folder, '{}.{}'.format(filename, ext)), 'wb') as file:
        pickle.dump(stuff, file)


def load(filename, ext='pkl', folder='data'):
    with open(path.join(folder, '{}.{}'.format(filename, ext)), 'rb') as file:
        return pickle.load(file)


def load_or_create(filename, ext='pkl', create_fn=None, folder='data', with_status=False):
    try:
        magic_obj = load(filename, ext, folder)
        status = True
    except FileNotFoundError:
        status = False
        if create_fn is None:
            magic_obj = None
        else:
            magic_obj = create_fn()
            save(magic_obj, filename, ext, folder)
    if with_status:
        return status, magic_obj
    else:
        return magic_obj

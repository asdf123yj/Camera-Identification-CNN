import glob


def load_img_from_dir(dir_path: str, f: str) -> [str]:
    if dir_path == '':
        return glob.glob('*')
    else:
        if dir_path[-1] != '/':
            dir_path += '/'
        return list(glob.glob(dir_path + '*.' + f))

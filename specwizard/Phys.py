import specwizard

def ReadPhys():
    import json
    path = specwizard.__file__[:-11]
    with open('{0}/Phys.data'.format(path), 'r') as f:
        x = json.load(f)
    return x

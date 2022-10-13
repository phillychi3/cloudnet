import numpy as gnp



def device(type='cpu'):
    global gnp
    if type.lower() == 'gpu':
        try:
            import cupy as gnp
        except Exception as e:
            print(e)
            print('cupy not found, using numpy')
            import numpy as gnp
    else:
        import numpy as gnp
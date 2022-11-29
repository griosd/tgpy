import torch
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from .tensor import to_numpy

__i_color = 0
_palette = sb.color_palette()


class DictObj(dict):
    """This class implements a dictionary object from a python dictionary."""
    def __init__(self, data=None, *args, **kwargs):
        """
        Initialization.

        :param data: a dictionary to be stored.
        :param args: arguments to be passed to the dict class.
        :param kwargs: keyword arguments to be passed to the dict class.
        """
        # noinspection PyArgumentList
        super().__init__(*args, **kwargs)
        if data is not None:
            for k, v in data.items():
                self[k] = v

    def __getattribute__(self, name):
        """
        Implements the __getattribute__ method.

        :param name: a string, the name of the attribute.

        :return: The value of the attribute, if the attribute exists on self, or call super.__getattribute__.
        """
        if name in self:
            return self[name]
        else:
            return super(DictObj, self).__getattribute__(name)

    def __getattr__(self, name):
        """
        Implements the __getattr__ method.

        :param name: a string, the name of the attribute.

        :return: The value of the attribute, if the attribute exists, or an AttributeError exception.
        """
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        """
        Implements the __setattr__ method.

        :param name: a string, the name of the attribute.
        :param value: the value assigned to the attribute.
        """
        if hasattr(self, name):
            super(DictObj, self).__setattr__(name, value)
        self[name] = value

    def __delattr__(self, name):
        """
        Implements the __delattr__ method.

        :param name: a string, the name of the attribute to be deleted.
        """
        if hasattr(self, name):
            super(DictObj, self).__delattr__(name)
        elif name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __repr__(self):
        """Implements (and returns) the __repr__ method, that is, the object representation."""
        types = [int, str, float, bool, tuple, dict, list]
        return type(self).__name__ + ':\n' + {k: v if ((type(v) in types) or (v is None)) else type(v)
                                              for k, v in self.items()}.__repr__().replace(', ', '\n')[1:-1]

    def print(self):
        """Returns the object representation with its elements."""
        print(type(self).__name__)
        for k, v in self.items():
            print(k, ':', v)

    def clone(self):
        """Returns a copy of the DictObj object."""
        return DictObj(data=self)

    def copy(self):
        """Returns a copy of the DictObj object."""
        return DictObj(data=self)

    def set_skip(self, key, value):
        """Implements the __setattr__ method if key don't exist."""
        if key != '__class__' and key not in self:
            self[key] = value


def plot2d(tensor1, tensor2=None, *args, **kwargs):
    if tensor2 is None:
        plt.plot(to_numpy(tensor1).T, *args, **kwargs)
    else:
        plt.plot(to_numpy(tensor1), to_numpy(tensor2), *args, **kwargs)


def color_next():
    """Returns the next color in the global palette _palette."""
    global __i_color
    r = color(__i_color)
    __i_color += 1
    return r


def color(i, palette=_palette):
    """
    Returns the i-th color of the palette, mod the size of the palette.

    :param i: an int.
    :param palette: a palette of colors.

    :return: the i-th color of the palette, mod the size of the palette.
    """
    return palette[i % len(palette)]
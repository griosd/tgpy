import os
import os.path
import sys
import traceback
import types
import weakref
import inspect
from torch import Tensor
from IPython.core.display import display
from importlib import import_module
from importlib.util import source_from_cache
from IPython.core.magic import Magics, magics_class, line_magic
from importlib import reload as _reload
skip_doctest = True


# ------------------------------------------------------------------------------
# Autoreload functionality
# ------------------------------------------------------------------------------
class ModuleReloader(object):
    enabled = False
    """Whether this reloader is enabled"""

    check_all = True
    """Autoreload all modules, not just those listed in 'modules'"""

    def __init__(self):
        # Modules that failed to reload: {module: mtime-on-failed-reload, ...}
        self.failed = {}
        # Modules specially marked as autoreloadable.
        self.modules = {}
        # Modules specially marked as not autoreloadable.
        self.skip_modules = {}
        # (module-name, name) -> weakref, for replacing old code objects
        self.old_objects = {}
        # Module modification timestamps
        self.modules_mtimes = {}

        # Cache module modification times
        self.check(check_all=True, do_reload=False)

    def mark_module_skipped(self, module_name):
        """Skip reloading the named module in the future"""
        try:
            del self.modules[module_name]
        except KeyError:
            pass
        self.skip_modules[module_name] = True

    def mark_module_reloadable(self, module_name):
        """Reload the named module in the future (if it is imported)"""
        try:
            del self.skip_modules[module_name]
        except KeyError:
            pass
        self.modules[module_name] = True

    def aimport_module(self, module_name):
        """Import a module, and mark it reloadable

        Returns
        -------
        top_module : module
            The imported module if it is top-level, or the top-level
        top_name : module
            Name of top_module

        """
        self.mark_module_reloadable(module_name)

        import_module(module_name)
        top_name = module_name.split('.')[0]
        top_module = sys.modules[top_name]
        return top_module, top_name

    @staticmethod
    def filename_and_mtime(module):
        if not hasattr(module, '__file__') or module.__file__ is None:
            return None, None

        if getattr(module, '__name__', None) in [None, '__mp_main__', '__main__']:
            # we cannot reload(__main__) or reload(__mp_main__)
            return None, None

        filename = module.__file__
        path, ext = os.path.splitext(filename)

        if ext.lower() == '.py':
            py_filename = filename
        else:
            try:
                py_filename = source_from_cache(filename)
            except ValueError:
                return None, None

        try:
            pymtime = os.stat(py_filename).st_mtime
        except OSError:
            return None, None

        return py_filename, pymtime

    def check(self, check_all=False, do_reload=True):
        """Check whether some modules need to be reloaded."""

        if not self.enabled and not check_all:
            return

        if check_all or self.check_all:
            modules = list(sys.modules.keys())
        else:
            modules = list(self.modules.keys())

        for modname in modules:
            m = sys.modules.get(modname, None)

            if modname in self.skip_modules:
                continue

            py_filename, pymtime = self.filename_and_mtime(m)
            if py_filename is None:
                continue

            try:
                if pymtime <= self.modules_mtimes[modname]:
                    continue
            except KeyError:
                self.modules_mtimes[modname] = pymtime
                continue
            else:
                if self.failed.get(py_filename, None) == pymtime:
                    continue

            self.modules_mtimes[modname] = pymtime

            # If we've reached this point, we should try to reload the module
            if do_reload:
                try:
                    superreload(m, _reload, self.old_objects)
                    if py_filename in self.failed:
                        del self.failed[py_filename]
                except (TypeError, AttributeError, KeyError):
                    print("[autoreload of %s failed: %s]" % (
                        modname, traceback.format_exc(10)), file=sys.stderr)
                    self.failed[py_filename] = pymtime


# ------------------------------------------------------------------------------
# superreload
# ------------------------------------------------------------------------------


func_attrs = ['__code__', '__defaults__', '__doc__',
              '__closure__', '__globals__', '__dict__']


def update_function(old, new):
    """Upgrade the code object of a function"""
    for name in func_attrs:
        try:
            setattr(old, name, getattr(new, name))
        except (AttributeError, TypeError):
            pass


def update_instances(old, new, objects=None, visited=None):
    """Iterate through objects recursively, searching for instances of old and
    replace their __class__ reference with new. If no objects are given, start
    with the current ipython workspace.
    """
    visited = visited if visited else {}
    if objects is None:
        # make sure visited is cleaned when not called recursively
        visited = {}
        # find ipython workspace stack frame
        frame = next(frame_nfo.frame for frame_nfo in inspect.stack()
                     if 'trigger' in frame_nfo.function)
        # build generator for non-private variable values from workspace
        shell = frame.f_locals['self'].shell
        user_ns = shell.user_ns
        user_ns_hidden = shell.user_ns_hidden
        non_matching = object()
        objects = (value for key, value in user_ns.items() if not key.startswith('_') and not
                   type(key) in (float, str) and (value is not user_ns_hidden.get(key, non_matching)) and not
                   inspect.ismodule(value))

    # use dict values if objects is a dict but don't touch private variables
    if hasattr(objects, 'items'):
        objects = (value for key, value in objects.items()
                   if not str(key).startswith('_') and not type(key) in (float, str) and not inspect.ismodule(value))

    # try if objects is iterable
    try:
        for obj in (obj for obj in objects if id(obj) not in visited):
            print(type(obj))
            # add current object to visited to avoid revisiting
            visited.update({id(obj): obj})

            # update, if object is instance of old_class (but no subclasses)
            if type(obj) is old:
                obj.__class__ = new

            # if object is instance of other class, look for nested instances
            if hasattr(obj, '__dict__') and not (inspect.isfunction(obj) or inspect.ismethod(obj)):
                update_instances(old, new, obj.__dict__, visited)

            # if object is a container, search it
            contains = hasattr(obj, '__contains__') and not \
                isinstance(obj, str) and not isinstance(obj, Tensor)
            if hasattr(obj, 'items') or contains:
                update_instances(old, new, obj, visited)

    except TypeError:
        pass


def update_class(old, new):
    """Replace stuff in the __dict__ of a class, and upgrade
    method code objects, and add new methods, if any"""
    for key in list(old.__dict__.keys()):
        old_obj = getattr(old, key)
        try:
            new_obj = getattr(new, key)
            # explicitly checking that comparison returns True to handle
            # cases where `==` doesn't return a boolean.
            if (old_obj == new_obj) is True:
                continue
        except AttributeError:
            # obsolete attribute: remove it
            try:
                delattr(old, key)
            except (AttributeError, TypeError):
                pass
            continue

        if update_generic(old_obj, new_obj):
            continue

        try:
            setattr(old, key, getattr(new, key))
        except (AttributeError, TypeError):
            pass  # skip non-writable attributes

    for key in list(new.__dict__.keys()):
        if key not in list(old.__dict__.keys()):
            try:
                setattr(old, key, getattr(new, key))
            except (AttributeError, TypeError):
                pass  # skip non-writable attributes

    # update all instances of class
    update_instances(old, new)


def update_property(old, new):
    """Replace get/set/del functions of a property"""
    update_generic(old.fdel, new.fdel)
    update_generic(old.fget, new.fget)
    update_generic(old.fset, new.fset)


def isinstance2(a, b, typ):
    return isinstance(a, typ) and isinstance(b, typ)


UPDATE_RULES = [
    (lambda a, b: isinstance2(a, b, type),
     update_class),
    (lambda a, b: isinstance2(a, b, types.FunctionType),
     update_function),
    (lambda a, b: isinstance2(a, b, property),
     update_property),
]
UPDATE_RULES.extend([(lambda a, b: isinstance2(a, b, types.MethodType),
                      lambda a, b: update_function(a.__func__, b.__func__)),
                     ])


def update_generic(a, b):
    for type_check, update in UPDATE_RULES:
        if type_check(a, b):
            update(a, b)
            return True
    return False


class StrongRef(object):
    def __init__(self, obj):
        self.obj = obj

    def __call__(self):
        return self.obj


def superreload(module, reload=_reload, old_objects=None):
    """Enhanced version of the builtin reload function.

    superreload remembers objects previously in the module, and

    - upgrades the class dictionary of every old class in the module
    - upgrades the code object of every old function and method
    - clears the module's namespace before reloading

    """
    if old_objects is None:
        old_objects = {}

    # collect old objects in the module
    for name, obj in list(module.__dict__.items()):
        if not hasattr(obj, '__module__') or obj.__module__ != module.__name__:
            continue
        key = (module.__name__, name)
        try:
            old_objects.setdefault(key, []).append(weakref.ref(obj))
        except TypeError:
            pass

    # reload module
    old_dict = dict()
    try:
        # clear namespace first from old cruft
        old_dict = module.__dict__.copy()
        old_name = module.__name__
        module.__dict__.clear()
        module.__dict__['__name__'] = old_name
        module.__dict__['__loader__'] = old_dict['__loader__']
    except (TypeError, AttributeError, KeyError):
        pass

    try:
        module = reload(module)
    except (TypeError, AttributeError, KeyError):
        # restore module dictionary on failed reload
        module.__dict__.update(old_dict)
        raise

    # iterate over all objects and update functions & classes
    for name, new_obj in list(module.__dict__.items()):
        key = (module.__name__, name)
        if key not in old_objects:
            continue

        new_refs = []
        for old_ref in old_objects[key]:
            old_obj = old_ref()
            if old_obj is None:
                continue
            new_refs.append(old_ref)
            update_generic(old_obj, new_obj)

        if new_refs:
            old_objects[key] = new_refs
        else:
            del old_objects[key]

    return module


# ------------------------------------------------------------------------------
# IPython connectivity
# ------------------------------------------------------------------------------


@magics_class
class AutoreloadMagics(Magics):
    def __init__(self, *a, **kwargs):
        # noinspection PyArgumentList
        super(AutoreloadMagics, self).__init__(*a, **kwargs)
        self._reloader = ModuleReloader()
        self._reloader.check_all = False
        self.loaded_modules = set(sys.modules)

    @line_magic
    def autoreload(self, parameter_s=''):
        r"""%autoreload => Reload modules automatically

        %autoreload
        Reload all modules (except those excluded by %aimport) automatically
        now.

        %autoreload 0
        Disable automatic reloading.

        %autoreload 1
        Reload all modules imported with %aimport every time before executing
        the Python code typed.

        %autoreload 2
        Reload all modules (except those excluded by %aimport) every time
        before executing the Python code typed.

        Reloading Python modules in a reliable way is in general
        difficult, and unexpected things may occur. %autoreload tries to
        work around common pitfalls by replacing function code objects and
        parts of classes previously in the module with new versions. This
        makes the following things to work:

        - Functions and classes imported via 'from xxx import foo' are upgraded
          to new versions when 'xxx' is reloaded.

        - Methods and properties of classes are upgraded on reload, so that
          calling 'c.foo()' on an object 'c' created before the reload causes
          the new code for 'foo' to be executed.

        Some of the known remaining caveats are:

        - Replacing code objects does not always succeed: changing a @property
          in a class to an ordinary method or a method to a member variable
          can cause problems (but in old objects only).

        - Functions that are removed (eg. via monkey-patching) from a module
          before it is reloaded are not upgraded.

        - C extension modules cannot be reloaded, and so cannot be
          autoreloaded.

        """
        if parameter_s == '':
            self._reloader.check(True)
        elif parameter_s == '0':
            self._reloader.enabled = False
        elif parameter_s == '1':
            self._reloader.check_all = False
            self._reloader.enabled = True
        elif parameter_s == '2':
            self._reloader.check_all = True
            self._reloader.enabled = True

    @line_magic
    def aimport(self, parameter_s='', stream=None):
        """%aimport => Import modules for automatic reloading.

        %aimport
        List modules to automatically import and not to import.

        %aimport foo
        Import module 'foo' and mark it to be autoreloaded for %autoreload 1

        %aimport foo, bar
        Import modules 'foo', 'bar' and mark them to be autoreloaded for %autoreload 1

        %aimport -foo
        Mark module 'foo' to not be autoreloaded for %autoreload 1
        """
        modname = parameter_s
        if not modname:
            to_reload = sorted(self._reloader.modules.keys())
            to_skip = sorted(self._reloader.skip_modules.keys())
            if stream is None:
                stream = sys.stdout
            if self._reloader.check_all:
                stream.write("Modules to reload:\nall-except-skipped\n")
            else:
                stream.write("Modules to reload:\n%s\n" % ' '.join(to_reload))
            stream.write("\nModules to skip:\n%s\n" % ' '.join(to_skip))
        elif modname.startswith('-'):
            modname = modname[1:]
            self._reloader.mark_module_skipped(modname)
        else:
            for _module in ([_.strip() for _ in modname.split(',')]):
                top_module, top_name = self._reloader.aimport_module(_module)

                # Inject module to user namespace
                self.shell.push({top_name: top_module})

    def pre_run_cell(self):
        if self._reloader.enabled:
            try:
                self._reloader.check()
            except (TypeError, AttributeError, KeyError, RecursionError):
                pass

    def post_execute_hook(self):
        """Cache the modification times of any modules imported in this execution."""
        newly_loaded_modules = set(sys.modules) - self.loaded_modules
        for modname in newly_loaded_modules:
            _, pymtime = self._reloader.filename_and_mtime(sys.modules[modname])
            if pymtime is not None:
                self._reloader.modules_mtimes[modname] = pymtime

        self.loaded_modules.update(newly_loaded_modules)


def autoreload(enable=True):
    """
    Enabled or disabled the autoreload

    :param enable: a boolean, indicate if enable or disable the autoreload.
    """
    if hasattr(__auto_reload, 'autoreload'):
        if enable:
            __auto_reload.autoreload('2')
            print('autoreload enabled')
        else:
            __auto_reload.autoreload('0')
            print('autoreload disabled')


def load_ipython_extension(shell):
    """
    Registers the skip magic when the extension loads.

    :param shell: a core.interactiveshell object from IPython.
    """
    try:
        global __auto_reload
        __auto_reload = AutoreloadMagics(shell)
        __auto_reload.autoreload('2')
        shell.register_magics(__auto_reload)
        shell.events.register('pre_run_cell', __auto_reload.pre_run_cell)
        shell.events.register('post_execute', __auto_reload.post_execute_hook)
    except Exception as e:
        display(e)


def unload_ipython_extension(shell):
    """
    Unregisters the skip magic when the extension unloads.

    :param shell: a core.interactiveshell object from IPython.
    """
    pass

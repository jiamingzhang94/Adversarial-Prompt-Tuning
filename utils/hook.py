import torch
import gc

class SingleModelHook():
    def __init__(self, model, name, use_inp=False, forward=True):
        self.model = model
        self.name = name
        _dict = {n: m for n, m in self.model.named_modules()}
        if self.name not in _dict.keys():
            raise NameError(f"No such name ({self.name}) in the model")

        self._module = _dict[self.name]
        self._hooked_value = None
        self.use_inp = use_inp
        self.forward = forward

        self._register_hook()

    def clear(self):
        """
        clear the hooked tensor
        """
        self._hooked_value = None

    def remove(self):
        self.handle.remove()

    def _register_hook(self):
        if self.use_inp:
            def hook(_, inp, __):
                if self._hooked_value is None:
                    self._hooked_value = inp[0]
                else:
                    self._hooked_value = torch.cat((self._hooked_value, inp[0]), dim=0)
        else:
            def hook(_, __, output):
                if self._hooked_value is None:
                    self._hooked_value = output
                else:
                    self._hooked_value = torch.cat((self._hooked_value, output), dim=0)

        if self.forward:
            self.handle = self._module.register_forward_hook(hook)
        else:
            self.handle = self._module.register_full_backward_hook(hook)

    def get_hooked_value(self):
        return self._hooked_value


class SingleModelHookCpu():
    def __init__(self, model, name, use_inp=False, forward=True):
        self.model = model
        self.name = name
        _dict = {n: m for n, m in self.model.named_modules()}
        if self.name not in _dict.keys():
            raise NameError(f"No such name ({self.name}) in the model")

        self._module = _dict[self.name]
        self._hooked_value = None
        self.use_inp = use_inp
        self.forward = forward

        self._register_hook()

    def clear(self):
        """
        clear the hooked tensor
        """
        self._hooked_value = None

    def _register_hook(self):
        if self.use_inp:
            def hook(_, inp, __):
                if self._hooked_value is None:
                    self._hooked_value = inp[0].cpu()
                else:
                    self._hooked_value = torch.cat((self._hooked_value, inp[0].cpu()), dim=0)
        else:
            def hook(_, __, output):
                if self._hooked_value is None:
                    self._hooked_value = output.cpu()
                else:
                    self._hooked_value = torch.cat((self._hooked_value, output.cpu()), dim=0)

        if self.forward:
            self.handle = self._module.register_forward_hook(hook)
        else:
            self.handle = self._module.register_full_backward_hook(hook)

    def get_hooked_value(self):
        return self._hooked_value

    def remove(self):
        self.handle.remove()

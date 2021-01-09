def check_layer_does_not_exit(self, model, layer) -> None:
    for name, child in model._modules.items():
        if child is not None:
            self.assertNotIsInstance(child, layer)
            check_layer_does_not_exit(self, child, layer)


def check_layer_exists(self, model, layer) -> None:
    for name, child in model._modules.items():
        if child is not None:
            self.assertNotIsInstance(child, layer)
            check_layer_exists(self, child, layer)

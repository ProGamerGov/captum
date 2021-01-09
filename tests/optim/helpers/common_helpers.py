def check_layer_not_in_model(self, model, layer) -> None:
    for name, child in model._modules.items():
        if child is not None:
            self.assertNotIsInstance(child, layer)
            check_layer_not_in_model(self, child, layer)

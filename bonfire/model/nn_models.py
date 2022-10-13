from abc import ABC

import torch

from bonfire.model.model_base import MultipleInstanceModel


class MultipleInstanceNN(MultipleInstanceModel, ABC):

    def forward(self, model_input):
        # We don't care about any interpretability output here
        bag_predictions, _ = self.forward_verbose(model_input)
        return bag_predictions

    def forward_verbose(self, model_input):
        unbatched_bag = False
        # Model input is tensor
        if torch.is_tensor(model_input):
            input_shape = model_input.shape
            unbatched_bag = len(input_shape) == self.n_expec_dims
            if unbatched_bag:
                # Just a single bag on its own, not in a batch, therefore stick it in a list
                bags = [model_input]
            else:
                # Assume already batched
                bags = model_input
        # Model input is list
        elif type(model_input) == list:
            # Assume already batched
            bags = model_input
        # Invalid input type
        else:
            raise ValueError('Invalid model input type {:}'.format(type(model_input)))

        # Actually pass the input through the model
        #  Note instance interpretations has a different meaning depending on the model
        #  They can be, but are not always, instance predictions (e.g., attention).
        bag_predictions, instance_interpretations = self._internal_forward(bags)

        # If given input was not batched, also un-batch the output
        if unbatched_bag:
            return bag_predictions[0], instance_interpretations[0]
        return bag_predictions, instance_interpretations


class InstanceSpaceNN(MultipleInstanceNN, ABC):

    name = "InstanceSpaceNN"

    def __init__(self, device, n_classes, n_expec_dims, encoder, instance_classifier, aggregation_func):
        super().__init__(device, n_classes, n_expec_dims)
        self.encoder = encoder
        self.instance_classifier = instance_classifier
        self.aggregation_func = aggregation_func

    def _internal_forward(self, bags):
        batch_size = len(bags)
        bag_predictions = torch.zeros((batch_size, self.n_classes)).to(self.device)
        # Bags may be of different sizes, so we can't use a tensor to store the instance predictions
        bag_instance_predictions = []
        for i, instances in enumerate(bags):
            # Embed instances
            instances = instances.to(self.device)
            instance_embeddings = self.encoder(instances)

            # Classify instances and aggregate
            #  Here, the instance interpretations are actually predictions
            instance_predictions = self.instance_classifier(instance_embeddings)
            bag_prediction = self.aggregation_func(instance_predictions)

            # Update outputs
            bag_predictions[i] = bag_prediction
            bag_instance_predictions.append(instance_predictions)
        return bag_predictions, bag_instance_predictions


class EmbeddingSpaceNN(MultipleInstanceNN, ABC):

    name = "EmbeddingSpaceNN"

    def __init__(self, device, n_classes, n_expec_dims, encoder, embedding_classifier, aggregation_func):
        super().__init__(device, n_classes, n_expec_dims)
        self.encoder = encoder
        self.embedding_classifier = embedding_classifier
        self.aggregation_func = aggregation_func

    def _internal_forward(self, bags):
        bag_predictions = torch.zeros((len(bags), self.n_classes)).to(self.device)
        for i, instances in enumerate(bags):
            # Embed instances
            instances = instances.to(self.device)
            instance_embeddings = self.encoder(instances)

            # Classify instances and aggregate
            #  This model does not produce any instance interpretations
            bag_embedding = self.aggregation_func(instance_embeddings)
            bag_prediction = self.embedding_classifier(bag_embedding)

            # Update outputs
            bag_predictions[i] = bag_prediction
        return bag_predictions, None

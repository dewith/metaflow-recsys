"""Flow for deploying the model in a sagemaker endpoint."""

import os

from metaflow import S3, FlowSpec, Parameter, current, step
from utils.config import get_parameters
from utils.logging import bprint


class DeploymentFlow(FlowSpec):
    """
    The flow for deploying the model in a sagemaker endpoint.

    Attributes
    ----------
    sage_defaults : dict
        The default parameters for the sagemaker endpoint.
    SAGEMAKER_DEPLOY : bool
        Deploy the model with Sagemaker.
    SAGEMAKER_IMAGE : str
        The image to use in the Sagemaker endpoint.
    SAGEMAKER_INSTANCE : str
        The AWS instance for the Sagemaker endpoint.
    SAGEMAKER_ROLE : str
        The AWS role for the Sagemaker endpoint.

    Methods
    -------
    start()
        The starting step of the flow.
    build()
        Step that takes the embedding space, build a Keras KNN model and store it in S3
        so that it can be deployed by a Sagemaker endpoint.
    deploy()
        Step to construct a SageMaker's TensorFlowModel from the tar in S3 and
        deploy it to a SageMaker endpoint.
    test()
        Step to check the SageMaker endpoint is working properly.
    end()
        The ending step of the flow.
    upload_model()
        Upload the model to S3.
    _get_flow_parameters()
        Get the class parameters for the flow.
    """

    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=import-outside-toplevel
    # pylint: disable=too-many-instance-attributes

    # Parameters
    _sage_defaults = get_parameters('sagemaker')

    SAGEMAKER_DEPLOY = Parameter(
        name='sagemaker_deploy',
        help='Deploy KNN model with Sagemaker',
        default=False,
        type=bool,
    )

    SAGEMAKER_IMAGE = Parameter(
        name='sagemaker_image',
        help='Image to use in the Sagemaker endpoint.',
        default=_sage_defaults['image'],
    )

    SAGEMAKER_INSTANCE = Parameter(
        name='sagemaker_instance',
        help='AWS instance for the Sagemaker endpoint.',
        default=_sage_defaults['instance'],
    )

    SAGEMAKER_ROLE = Parameter(
        name='sagemaker_role',
        help='IAM role in AWS to use to spin up the Sagemaker endpoint.',
        default=_sage_defaults['role'],
    )

    @step
    def start(self):
        """Start the flow and print a cool message."""
        bprint("🌀 Let's get started")
        bprint(f'Running: {current.flow_name} @ {current.run_id}')
        bprint(f'User: {current.username}')
        bprint('Parameters:')
        for k, v in self._get_flow_parameters().items():
            bprint(f'{k}: {v}', level=1)
        self.next(self.build)

    @step
    def build(self):
        """
        Take the embedding space, build a Keras KNN model and store it in S3
        so that it can be deployed by a Sagemaker endpoint.
        """
        import tarfile
        import time
        from pathlib import Path

        import numpy as np
        from model import RetrievalModel
        from utils.meta import get_latest_successful_run

        bprint('Building model', level=1)
        latest_run = get_latest_successful_run('ModelingFlow')
        self.final_vectors = latest_run.data.final_vectors
        self.songs_ids = np.array(self.final_vectors.index_to_key)
        self.songs_embeddings = np.array(
            [self.final_vectors[idx] for idx in self.songs_ids]
        )
        # First build the retrieval model and test it
        model = RetrievalModel(self.songs_ids, self.songs_embeddings)
        model.test()

        bprint('Saving model locally', level=2)
        self.model_ts = int(round(time.time() * 1000))  # Signature for the endpoint
        models_dir = Path('data/04_models')
        model_name = models_dir / f'playlist-recs-model-{self.model_ts}/1'
        tar_name = models_dir / f'model-{self.model_ts}.tar.gz'
        bprint(f'Model path: {model_name}', level=3)
        bprint(f'Tarfile path: {tar_name}', level=3)

        # Save the tfrs index model
        model.save(filepath=str(model_name))

        # Zip keras folder to a single tar local file
        with tarfile.open(tar_name, mode='w:gz') as _tar:
            _tar.add(model_name, recursive=True)

        # Upload the model
        bprint('Uploading model tar to S3', level=3)
        self.model_local_path = str(model_name)
        self.tar_local_path = str(tar_name)
        self.model_s3_path = self.upload_model(self.tar_local_path)
        bprint(f'Model saved at {self.model_s3_path}', level=4)

        self.next(self.deploy)

    @step
    def deploy(self):
        """
        Use SageMaker to deploy the model as a stand-alone, PaaS endpoint, with o
        ur choice of the underlying Docker image and hardware capabilities.
        """
        bprint('Deploying the model', level=1)

        if self.SAGEMAKER_DEPLOY:
            from sagemaker.deserializers import JSONDeserializer
            from sagemaker.serializers import JSONSerializer
            from sagemaker.tensorflow import TensorFlowModel

            bprint('Deploying the model to SageMaker', level=2)
            self.endpoint_name = f'playlist-recs-{self.model_ts}-endpoint'
            bprint(f'Endpoint name: {self.endpoint_name}', level=3)
            model = TensorFlowModel(
                model_data=self.model_s3_path,
                image_uri=self.SAGEMAKER_IMAGE,
                role=self.SAGEMAKER_ROLE,
            )
            model.deploy(
                initial_instance_count=1,
                instance_type=self.SAGEMAKER_INSTANCE,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer(),
                endpoint_name=self.endpoint_name,
            )
            bprint('Endpoint deployed', level=3)
        else:
            bprint('Skipping deployment to SageMaker', level=2)

        self.next(self.check)

    @step
    def check(self):
        """Test the endpoint and delete it."""
        import json
        import random

        from sagemaker.predictor import Predictor

        bprint('Testing the endpoint', level=1)

        test_index = random.randint(0, 10)
        test_id = self.songs_ids[test_index]
        data = json.dumps({'instances': [test_id]})
        bprint('Request body:', data, level=2)

        # Output looks like {'predictions': {'output_2': ['Pitbull-Timber', ..]}
        predictor = Predictor(self.endpoint_name)
        predictor.content_type = 'application/json'
        result = json.loads(predictor.predict(data))
        bprint('Response:', json.dumps(result, indent=4), level=2)
        bprint('Endpoint working!', level=2)

        bprint('Deleting endpoint now', level=2)
        predictor.delete_endpoint()
        bprint('Endpoint deleted!', level=3)
        self.next(self.end)

    @step
    def end(self):
        """End the flow and print a cool message."""
        bprint('✨ All done ᕙ(⇀‸↼‶)ᕗ')

    def upload_model(self, local_tar_name: str) -> str:
        """
        Upload the model to S3 in the Sagemaker endpoint.

        Parameters
        ----------
        local_tar_name : str
            The path to the local tar file.

        Returns
        -------
        str
            The S3 URL of the uploaded model.
        """
        # Metaflow s3 client needs a byte object for the put
        with open(local_tar_name, 'rb') as in_file:
            data = in_file.read()
            with S3(run=self) as s3:
                s3_url = s3.put(local_tar_name, data)
        return s3_url

    def _get_flow_parameters(self):
        """
        Get the parameters of the flow while skipping the private ones.

        Returns
        -------
        dict
            The parameters of the flow as a dictionary.
        """
        parameters = {}
        for attr_name, attr_value in self.__class__.__dict__.items():
            if not callable(attr_value) and not attr_name.startswith('_'):
                pretty_attr_name = attr_name.replace('_', ' ').capitalize()
                parameters[pretty_attr_name] = getattr(self, attr_name)
        return parameters


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    DeploymentFlow()

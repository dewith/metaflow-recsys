"""Flow for deploying the model in a sagemaker endpoint."""

import os
import tarfile
import time
from pathlib import Path

from metaflow import S3, FlowSpec, Parameter, current, step
from model import RetrievalModel
from sagemaker.tensorflow import TensorFlowModel
from utils.config import get_parameters
from utils.logging import bprint
from utils.meta import get_latest_successful_run


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
    build_retrieval_model()
        Take the embedding space, build a Keras KNN model and store it in S3
        so that it can be deployed by a Sagemaker endpoint.
    start()
        The starting step of the flow.
    deploy()
        Train the track2vec model on the train dataset.
    end()
        End the flow and print a cool message.
    """

    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=import-outside-toplevel
    # pylint: disable=too-many-instance-attributes

    # Parameters
    sage_defaults = get_parameters('sagemaker')

    SAGEMAKER_DEPLOY = Parameter(
        name='sagemaker_deploy',
        help='Deploy KNN model with Sagemaker',
        default=False,
        type=bool,
    )

    SAGEMAKER_IMAGE = Parameter(
        name='sagemaker_image',
        help='Image to use in the Sagemaker endpoint.',
        default=sage_defaults['image'],
    )

    SAGEMAKER_INSTANCE = Parameter(
        name='sagemaker_instance',
        help='AWS instance for the Sagemaker endpoint.',
        default=sage_defaults['instance'],
    )

    SAGEMAKER_ROLE = Parameter(
        name='sagemaker_role',
        help='IAM role in AWS to use to spin up the Sagemaker endpoint.',
        default=sage_defaults['role'],
    )

    def build_retrieval_model(self):
        """
        Take the embedding space, build a Keras KNN model and store it in S3
        so that it can be deployed by a Sagemaker endpoint.

        While for simplicity this function is embedded in the deploy step,
        you could think of spinning it out as it's own step.

        Returns
        -------
        str
            The path to the model tarfile.
        """

        model = RetrievalModel(self.songs_ids, self.songs_embeddings)
        model.test()

        bprint('Saving model locally', level=2)
        # Signature for the endpointand timestamp as a convention
        model_timestamp = int(round(time.time() * 1000))

        # TF models need to have a version
        models_dir = Path('data/04_models')
        model_name = models_dir / f'playlist-recs-model-{model_timestamp}/1'
        local_tar_name = models_dir / f'model-{model_timestamp}.tar.gz'
        bprint(f'Model path: {model_name}', level=3)
        bprint(f'Tarfile path: {local_tar_name}', level=3)

        # Save the tfrs index model
        model.save(filepath=str(model_name))

        # Zip keras folder to a single tar local file
        with tarfile.open(local_tar_name, mode='w:gz') as _tar:
            _tar.add(model_name, recursive=True)

        return local_tar_name

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

    @step
    def start(self):
        """Start the flow and print a cool message."""
        bprint("🌀 Let's get started")
        bprint(f'Running: {current.flow_name} @ {current.run_id}')
        bprint(f'User: {current.username}')
        self.next(self.deploy)

    @step
    def deploy(self):
        """
        Use SageMaker to deploy the model as a stand-alone, PaaS endpoint, with o
        ur choice of the underlying Docker image and hardware capabilities.
        """
        import numpy as np

        bprint('Deploying the model', level=1)
        latest_run = get_latest_successful_run('ModelingFlow')
        self.final_vectors = latest_run.data.final_vectors
        self.songs_ids = np.array(self.final_vectors.index_to_key)
        self.songs_embeddings = np.array(
            [self.final_vectors[idx] for idx in self.songs_ids]
        )
        # First build the retrieval model and save it locally
        local_tar_path = self.build_retrieval_model()

        if self.SAGEMAKER_DEPLOY:
            bprint('Deploying the model to SageMaker', level=2)

            # Upload the model
            bprint('Uploading model tar to S3', level=3)
            self.model_s3_path = self.upload_model(local_tar_path)
            bprint(f'Model saved at {self.model_s3_path}', level=4)

            # Deploy the model
            bprint('Deploying the endpoint', level=3)
            self.endpoint_name = 'playlist-recs-{self.model_timestamp}-endpoint'
            model = TensorFlowModel(
                model_data=self.model_s3_path,
                image_uri=self.SAGEMAKER_IMAGE,
                role=self.SAGEMAKER_ROLE,
            )
            predictor = model.deploy(
                initial_instance_count=1,
                instance_type=self.SAGEMAKER_INSTANCE,
                endpoint_name=self.endpoint_name,
            )
            bprint(f'Endpoint name: {self.endpoint_name}', level=4)

            # Test against the endpoint
            # Output looks like {'predictions': {'output_2': ['0012E00001z5EzAQAU', ..]}
            bprint('Running a test against the endpoint', level=3)
            input_ = {'instances': np.array([self.all_ids[self.test_index]])}
            result = predictor.predict(input_)
            bprint(f'Test input: {input_}', level=4)
            bprint(f'Test result: {result}', level=4)
            time.sleep(2)

            # Delete the endpoint. Sagemaker is expensive :(
            bprint('Deleting endpoint now...', level=3)
            predictor.delete_endpoint()
            bprint('Endpoint deleted!', level=4)

        self.next(self.end)

    @step
    def end(self):
        """End the flow and print a cool message."""
        bprint('✨ All done ᕙ(⇀‸↼‶)ᕗ')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    DeploymentFlow()

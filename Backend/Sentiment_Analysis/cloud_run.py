from google.cloud.aiplatform_v1.types import PredictRequest
from google.protobuf import struct_pb2
from google.cloud import aiplatform
from typing import Union, Dict, List


import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/captain/Downloads/bert-vertex-ai-203e20614649.json"

def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    # Ensure instances is a list of dictionaries
    instances = instances if isinstance(instances, list) else [instances]

    # Convert Python dicts to protobuf `Value`
    protobuf_instances = []
    for instance in instances:
        struct_val = struct_pb2.Value()
        struct_val.struct_value.update(instance)
        protobuf_instances.append(struct_val)

    parameters = None

    endpoint_path = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )

    response = client.predict(
        endpoint=endpoint_path,
        instances=instances,
        parameters=parameters
    )

    return response.predictions

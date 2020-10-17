import os
from time import sleep

import pytest
import docker
import requests

@pytest.mark.integration
def test_predict_endpoint():
    base_path = os.path.abspath(os.path.join(__file__, '../../'))
    client = docker.from_env()
    image, _ = client.images.build(path=base_path)
    container = client.containers.run(image,
                                      detach=True,
                                      ports={'5000/tcp': ('127.0.0.1', 5000)})

    sleep(5)

    r = requests.post("http://127.0.0.1:5000/predict",
                      json={'data': [[0, 1], [2, 3]]})

    assert r.text == '{\n  "prediction": [\n    "[0.30583531]", \n    "[1.77134556]"\n  ]\n}\n'
    container.stop()


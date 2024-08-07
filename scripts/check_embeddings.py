import requests
import numpy as np
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    input_texts = ['query: how much protein should a female eat']

    model = SentenceTransformer("intfloat/multilingual-e5-large")
    embeddings = model.encode(input_texts, normalize_embeddings=True)
    print(embeddings)
    response = requests.post(
        "http://triton-ingress-controller.triton.k8s.ml-infra-xc/triton-multilingual-e5-large/v2/models/wb_embedder_v0/infer",
        json={"inputs": [
            {
                "name": "INPUT_TEXT",
                "datatype": "BYTES",
                "shape": [len(input_texts), 1],
                "data": input_texts,
            },
        ]}
    ).json()
    triton_vector = np.array((response["outputs"][0]["data"]))
    print(triton_vector.reshape(-1, 1024))


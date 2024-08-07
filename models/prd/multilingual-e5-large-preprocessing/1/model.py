from transformers import AutoTokenizer

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self._logger = pb_utils.Logger
        # You must parse model_config. JSON string is not parsed here
        model_path = pb_utils.get_model_dir()
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large", use_fast=True)
        self._logger.log_info("Successfull init!")
        self._logger.log_info(f"Using fast version of tokenizer on RUST: {self.tokenizer.is_fast}")

    def execute(self, requests):
        responses = []

        for request in requests:
            input_ = pb_utils.get_input_tensor_by_name(request, "text")
            input_string = input_.as_numpy()
            requests_texts = [i[0].decode("utf-8") for i in input_string]

            batch = self.tokenizer(
                requests_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="np",
            )
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "output_input_ids",
                            batch["input_ids"],
                        ),
                        pb_utils.Tensor(
                            "output_attention_mask",
                            batch["attention_mask"],
                        )
                    ]
                )
            )

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        self.tokenizer = None


import json
import logging
import copy
from abc import ABC

from api.models.utils import logger,element_to_dict
from api.schema import EmbeddingsRequest


from oci.generative_ai_inference import models as oci_models
from oci.generative_ai_inference import GenerativeAiInferenceClient

from api.setting import (
    DEBUG,
    CLIENT_KWARGS,
    INFERENCE_ENDPOINT_TEMPLATE,
    SUPPORTED_OCIGENAI_EMBEDDING_MODELS,
    OCI_REGION,
    OCI_COMPARTMENT
)


from fastapi import HTTPException

from api.models.base import BaseEmbeddingsModel
from api.schema import (
    EmbeddingsRequest,
    EmbeddingsResponse
)
from api.models.adapter.embed_adapter import EmbedRequestAdapter
from api.models.utils import logger




class OCIGenAIEmbeddingsModel(BaseEmbeddingsModel, ABC):
    def __init__(self):
        self.generative_ai_inference_client = GenerativeAiInferenceClient(**CLIENT_KWARGS)
        self.init_models()

    def init_models(self):
        """Initialize embedding models from OCI API if not loaded from yaml"""
        if not SUPPORTED_OCIGENAI_EMBEDDING_MODELS:
            from oci.generative_ai import GenerativeAiClient

            client_kwargs = CLIENT_KWARGS.copy()
            client_kwargs['service_endpoint'] = f"https://generativeai.{OCI_REGION}.oci.oraclecloud.com"
            generative_ai_client = GenerativeAiClient(**client_kwargs)

            list_models_response = generative_ai_client.list_models(
                compartment_id=OCI_COMPARTMENT,
                lifecycle_state="ACTIVE"
            )

            for model in list_models_response.data.items:
                if "TEXT_EMBEDDINGS" in model.capabilities:
                    model_name = model.display_name
                    SUPPORTED_OCIGENAI_EMBEDDING_MODELS[model_name] = {
                        "type": "embedding",
                        "name": model_name,
                        "model_id": model_name,
                        "provider": model_name.split(".")[0] if "." in model_name else "UNKNOWN",
                        "region": OCI_REGION,
                        "compartment_id": OCI_COMPARTMENT,
                    }

            logger.info(f"Successfully loaded {len(SUPPORTED_OCIGENAI_EMBEDDING_MODELS)} embedding models from API")

    def _log_chat(self,content,schema):
        def shortern(input):
            string = str(input)
            if string.startswith("data:image/"):
                return string[:30] + "..."
            elif string.startswith("["):
                return string[:20] + "..." +string[-20:]
            else:
                return string

        def modify_msg(input):
            if isinstance(input, str):
                return shortern(input)
            else:
                output = []
                for message in input:
                    if isinstance(message, dict):
                        output.append(shortern(message["embedding"]))
                    else:
                        output.append(shortern(message))
                return output
        try:
            temp_content = element_to_dict(copy.deepcopy(content))
            if schema == "raw_request":
                temp_content["input"] = modify_msg(temp_content["input"])
            elif schema == "oci_request":
                temp_content["inputs"] = modify_msg(temp_content["inputs"])
            elif schema == "oci_response":
                temp_content["embeddings"] = modify_msg(temp_content["embeddings"])
            elif schema == "raw_response":
                temp_content["data"] = modify_msg(temp_content["data"])
            
            return json.dumps(temp_content, indent=2,ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Failed to convert {schema}:"+str(e))
            return str(content)

    def _invoke_model(self, request: EmbeddingsRequest) -> oci_models.EmbedTextResult:
        if DEBUG:
            logger.info("Raw request:\n" + self._log_chat(request,"raw_request"))

        model_name = request.get("model")
        model_info = SUPPORTED_OCIGENAI_EMBEDDING_MODELS[model_name]
        region = model_info["region"]
        self.generative_ai_inference_client.base_client._endpoint = INFERENCE_ENDPOINT_TEMPLATE.replace("{region}", region)
        embed_text_details = EmbedRequestAdapter(model_info).to_oci(request)  
        if DEBUG:
            logger.info("OCI Generative AI request:\n" + self._log_chat(embed_text_details,"oci_request"))     
        
        try:
            response = self.generative_ai_inference_client.embed_text(
                embed_text_details = embed_text_details
                )
            embed_text_response = response.data
            if DEBUG:
                logger.info("OCI Generative AI response:\n" + self._log_chat(embed_text_response,"oci_response"))
            return embed_text_response
        except Exception as e:
            logger.error("Validation Error: " + str(e))
            raise HTTPException(status_code=400, detail=str(e))


    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        response = self._invoke_model(embeddings_request)
        embde_response = EmbedRequestAdapter.to_openai(response)
        if DEBUG:
            logger.info("Raw response:\n" + self._log_chat(embde_response,"raw_response"))
        return embde_response


def get_embeddings_model(model_id: str) -> OCIGenAIEmbeddingsModel:
    # Create model instance first to trigger init_models() if needed
    model_instance = OCIGenAIEmbeddingsModel()

    model_info = SUPPORTED_OCIGENAI_EMBEDDING_MODELS.get(model_id, "")
    if model_info:
        if DEBUG:
            logger.info("model name is " + model_info["name"])
        return model_instance
    else:
        logger.error("Unsupported model id " + model_id)
        raise HTTPException(
            status_code=400,
            detail="Unsupported embedding model id " + model_id,
        )

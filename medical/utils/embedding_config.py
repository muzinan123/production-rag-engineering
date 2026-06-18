from dataclasses import dataclass
from enum import Enum
from typing import Optional

class EmbeddingProvider(Enum):
    BEDROCK = "bedrock"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"

@dataclass
class EmbeddingConfig:
    provider: EmbeddingProvider
    model_name: str  
    aws_region: Optional[str] = None

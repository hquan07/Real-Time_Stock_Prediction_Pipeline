import io
import os
import json
from fastavro import parse_schema, schemaless_writer

from ..config.kafka_config import AVRO_SCHEMA_PATH


class AvroSerializer:
    def __init__(self):
        if not os.path.exists(AVRO_SCHEMA_PATH):
            raise FileNotFoundError(f"Avro schema not found: {AVRO_SCHEMA_PATH}")

        with open(AVRO_SCHEMA_PATH, "r") as f:
            self.schema = parse_schema(json.load(f))

    def serialize(self, data: dict) -> bytes:
        """
        Serialize dict to Avro binary
        """
        bytes_writer = io.BytesIO()
        schemaless_writer(bytes_writer, self.schema, data)
        return bytes_writer.getvalue()
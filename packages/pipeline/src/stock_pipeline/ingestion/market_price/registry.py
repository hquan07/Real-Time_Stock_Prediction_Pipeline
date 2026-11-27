"""Confluent Schema Registry contract registration and writer-schema lookup."""

from __future__ import annotations

import json
from urllib.parse import quote

import requests

from .contract import SCHEMA, SUBJECT, VALIDATED_SUBJECT


class SchemaRegistry:
    def __init__(self, base_url: str, *, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def register_market_price(self) -> int:
        raw_id = None
        for name in (SUBJECT, VALIDATED_SUBJECT):
            subject = quote(name, safe="")
            config = requests.put(
                f"{self.base_url}/config/{subject}",
                json={"compatibility": "BACKWARD"}, timeout=self.timeout,
            )
            config.raise_for_status()
            response = requests.post(
                f"{self.base_url}/subjects/{subject}/versions",
                json={"schemaType": "AVRO", "schema": json.dumps(SCHEMA, separators=(",", ":"))},
                timeout=self.timeout,
            )
            response.raise_for_status()
            if name == SUBJECT:
                raw_id = int(response.json()["id"])
        return int(raw_id)

    def get_schema(self, schema_id: int) -> dict:
        response = requests.get(
            f"{self.base_url}/schemas/ids/{schema_id}", timeout=self.timeout,
        )
        if response.status_code == 404:
            raise ValueError("Unknown Schema Registry ID")
        response.raise_for_status()
        payload = response.json()
        if payload.get("schemaType", "AVRO") != "AVRO":
            raise ValueError("Unsupported schema type")
        return json.loads(payload["schema"])
